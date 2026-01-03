from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn

from litept.datasets.transform import Compose
from litept.models.litept import LitePT


class LitePTONNX(nn.Module):
    """
    Wrapper to present a stable ONNX I/O signature.

    Inputs:
      - feat:        (N, F) float32/float16
      - grid_coord:  (N, 3) int32/int64
      - offset:      (B,)   int64 cumulative counts (last element == N)

    Output:
      - feat_out:    (N_down, C) float32/float16
    """

    def __init__(self, model: LitePT) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        feat: torch.Tensor,
        grid_coord: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        data_dict = {
            "feat": feat,
            "grid_coord": grid_coord,
            "offset": offset,
        }
        point = self.model(data_dict)
        return point.feat


def _load_pretrained_backbone_weights(model: LitePT) -> None:
    """
    Optional: load the public NuScenes semantic segmentation pretrained checkpoint
    and map the keys to the backbone used by tools/demo.py.
    """
    ckpt_path = hf_hub_download(
        repo_id="prs-eth/LitePT",
        filename="nuscenes-semseg-litept-small-v1m1/model/model_best.pth",
        repo_type="model",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    weight = OrderedDict()
    prefix = "module.backbone."
    for key, value in ckpt["state_dict"].items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            weight[new_key] = value
    model.load_state_dict(weight, strict=True)


def _prepare_demo_sample(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build an input sample matching tools/demo.py preprocessing, but return only
    (feat, grid_coord, offset) needed for ONNX export.
    """
    lidar_path = hf_hub_download(
        repo_id="prs-eth/LitePT_demo",
        filename="outdoor_sample1.bin",
        repo_type="dataset",
        revision="main",
    )
    points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])
    coord = points[:, :3]  # [N, 3]
    strength = points[:, 3].reshape([-1, 1]) / 255.0  # [0, 1]

    point: dict[str, Any] = {"coord": coord, "strength": strength}

    data_config = [
        dict(
            type="GridSample",
            grid_size=0.05,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            # demo includes inverse; for export we only need grid_coord/feat/offset
            keys=("coord", "grid_coord"),
            feat_keys=("coord", "strength"),
        ),
    ]
    transform = Compose(data_config)
    point = transform(point)

    feat = point["feat"].to(device=device, dtype=dtype, non_blocking=True)
    grid_coord = point["grid_coord"].to(device=device, non_blocking=True)

    # Some pipelines produce "offset"; if not, assume batch size 1.
    if "offset" in point:
        offset = point["offset"].to(device=device, dtype=torch.int64, non_blocking=True)
    else:
        offset = torch.tensor([grid_coord.shape[0]], device=device, dtype=torch.int64)

    # Ensure predictable dtypes for TRT plugins / shape inference
    if grid_coord.dtype not in (torch.int32, torch.int64):
        grid_coord = grid_coord.to(torch.int32)

    return feat, grid_coord, offset


def _make_dummy_inputs(
    n_points: int,
    feat_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create random dummy inputs with batch size 1.
    """
    feat = torch.randn((n_points, feat_dim), device=device, dtype=dtype)
    grid_coord = torch.randint(0, 256, (n_points, 3), device=device, dtype=torch.int32)
    offset = torch.tensor([n_points], device=device, dtype=torch.int64)
    return feat, grid_coord, offset


def export_onnx(
    out_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    opset: int,
    do_constant_folding: bool,
    use_demo_sample: bool,
    n_points: int,
    feat_dim: int,
    load_pretrained: bool,
) -> None:
    if device.type == "cpu":
        raise RuntimeError(
            "CPU ONNX export is not supported for this LitePT build because it uses spconv "
            "(implicit GEMM) which requires CUDA. Re-run with --device cuda."
        )

    model = LitePT()
    if load_pretrained:
        _load_pretrained_backbone_weights(model)

    model.eval()
    model.to(device=device)

    wrapper = LitePTONNX(model).eval().to(device=device)

    if use_demo_sample:
        feat, grid_coord, offset = _prepare_demo_sample(device=device, dtype=dtype)
    else:
        feat, grid_coord, offset = _make_dummy_inputs(
            n_points=n_points,
            feat_dim=feat_dim,
            device=device,
            dtype=dtype,
        )

    # ONNX export settings
    input_names = ["feat", "grid_coord", "offset"]
    output_names = ["feat_out"]

    # Dynamic axes:
    # - N: number of points
    # - B: batch size in "offset" (cumulative counts)
    dynamic_axes = {
        "feat": {0: "N"},
        "grid_coord": {0: "N"},
        "offset": {0: "B"},
        "feat_out": {0: "N_down"},
    }

    # IMPORTANT:
    # - This relies on custom ops being represented via .symbolic() (custom domain like litept::PointROPE).
    # - TensorRT will need a plugin to handle those custom ops.
    torch.onnx.export(
        wrapper,
        (feat, grid_coord, offset),
        f=str(out_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
        # Keep initializers as inputs is typically False for TRT
        keep_initializers_as_inputs=False,
        verbose=False,
    )


def _parse_dtype(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s} (use fp16/fp32)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LitePT to ONNX (dynamic shapes, custom ops)."
    )
    parser.add_argument("--out", type=str, required=True, help="Output ONNX file path.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Export device. Note: spconv implicit GEMM requires CUDA; CPU export is not supported.",
    )
    parser.add_argument(
        "--dtype", type=str, default="fp16", help="Input/weight dtype: fp16 or fp32."
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--no-constant-folding",
        action="store_true",
        help="Disable constant folding (sometimes helps preserve custom ops / avoid folding issues).",
    )
    parser.add_argument(
        "--use-demo-sample",
        action="store_true",
        help="Use the same preprocessing as tools/demo.py (downloads a sample).",
    )
    parser.add_argument(
        "--n-points", type=int, default=32768, help="Dummy input: number of points."
    )
    parser.add_argument(
        "--feat-dim", type=int, default=4, help="Dummy input: feature dimension."
    )
    parser.add_argument(
        "--load-pretrained",
        action="store_true",
        help="Load public pretrained NuScenes semseg checkpoint (backbone weights).",
    )

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    if device.type == "cpu" and dtype == torch.float16:
        # CPU fp16 is generally not supported for many ops; fall back to fp32.
        dtype = torch.float32

    export_onnx(
        out_path=out_path,
        device=device,
        dtype=dtype,
        opset=int(args.opset),
        do_constant_folding=not args.no_constant_folding,
        use_demo_sample=bool(args.use_demo_sample),
        n_points=int(args.n_points),
        feat_dim=int(args.feat_dim),
        load_pretrained=bool(args.load_pretrained),
    )

    print(f"Exported ONNX to: {out_path}")


if __name__ == "__main__":
    main()
