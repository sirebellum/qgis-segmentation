# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Export helpers for runtime artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.runtime_numpy import RUNTIME_META_VERSION

from .data.dataset import UnsupervisedRasterDataset
from .data.synthetic import SyntheticDataset
from .config import Config
from .losses import total_loss
from .models.model import MonolithicSegmenter
from .utils.seed import set_seed


def _rename_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract and rename the subset of weights consumed by the numpy runtime.

    The runtime expects a flattened naming scheme (stem, block1/2, seed_proj).
    """
    mapping = {
        # Stem (nn.Sequential indices 0/1/3/4)
        "encoder.stem.0.weight": "stem.conv1.weight",
        "encoder.stem.0.bias": "stem.conv1.bias",
        "encoder.stem.1.weight": "stem.bn1.weight",
        "encoder.stem.1.bias": "stem.bn1.bias",
        "encoder.stem.1.running_mean": "stem.bn1.running_mean",
        "encoder.stem.1.running_var": "stem.bn1.running_var",
        "encoder.stem.3.weight": "stem.conv2.weight",
        "encoder.stem.3.bias": "stem.conv2.bias",
        "encoder.stem.4.weight": "stem.bn2.weight",
        "encoder.stem.4.bias": "stem.bn2.bias",
        "encoder.stem.4.running_mean": "stem.bn2.running_mean",
        "encoder.stem.4.running_var": "stem.bn2.running_var",
        # Residual blocks
        "encoder.block1.conv1.weight": "block1.conv1.weight",
        "encoder.block1.conv1.bias": "block1.conv1.bias",
        "encoder.block1.bn1.weight": "block1.bn1.weight",
        "encoder.block1.bn1.bias": "block1.bn1.bias",
        "encoder.block1.bn1.running_mean": "block1.bn1.running_mean",
        "encoder.block1.bn1.running_var": "block1.bn1.running_var",
        "encoder.block1.conv2.weight": "block1.conv2.weight",
        "encoder.block1.conv2.bias": "block1.conv2.bias",
        "encoder.block1.bn2.weight": "block1.bn2.weight",
        "encoder.block1.bn2.bias": "block1.bn2.bias",
        "encoder.block1.bn2.running_mean": "block1.bn2.running_mean",
        "encoder.block1.bn2.running_var": "block1.bn2.running_var",
        "encoder.block2.conv1.weight": "block2.conv1.weight",
        "encoder.block2.conv1.bias": "block2.conv1.bias",
        "encoder.block2.bn1.weight": "block2.bn1.weight",
        "encoder.block2.bn1.bias": "block2.bn1.bias",
        "encoder.block2.bn1.running_mean": "block2.bn1.running_mean",
        "encoder.block2.bn1.running_var": "block2.bn1.running_var",
        "encoder.block2.conv2.weight": "block2.conv2.weight",
        "encoder.block2.conv2.bias": "block2.conv2.bias",
        "encoder.block2.bn2.weight": "block2.bn2.weight",
        "encoder.block2.bn2.bias": "block2.bn2.bias",
        "encoder.block2.bn2.running_mean": "block2.bn2.running_mean",
        "encoder.block2.bn2.running_var": "block2.bn2.running_var",
        # Seed projection
        "cluster_head.seed_proj.weight": "seed_proj.weight",
        "cluster_head.seed_proj.bias": "seed_proj.bias",
    }
    renamed: Dict[str, torch.Tensor] = {}
    missing = []
    for old, new in mapping.items():
        tensor = state_dict.get(old)
        if tensor is None:
            missing.append(old)
            continue
        renamed[new] = tensor.detach().cpu()
    if missing:
        raise KeyError(f"Missing expected weights in checkpoint: {missing}")
    return renamed


def export_numpy_artifacts(
    model_state: Dict[str, torch.Tensor],
    cfg: Config,
    score: float,
    step: int,
    out_dir: str,
    mirror_dir: Optional[str] = None,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Write numpy weights + metadata for runtime consumption.

    Args:
        model_state: state_dict() of MonolithicSegmenter.
        cfg: loaded Config object (for embed_dim/max_k/etc.).
        score: scalar metric used for best-model selection.
        step: global step associated with the export.
        out_dir: primary export destination (e.g., model/best).
        mirror_dir: optional secondary destination (training/best_model).
        extra_metrics: optional additional metrics to persist in metrics.json.
    """
    primary = Path(out_dir)
    primary.mkdir(parents=True, exist_ok=True)
    if mirror_dir:
        Path(mirror_dir).mkdir(parents=True, exist_ok=True)

    weights = _rename_state_dict(model_state)
    weights_np = {k: v.numpy() for k, v in weights.items()}

    meta = {
        "version": RUNTIME_META_VERSION,
        "max_k": int(cfg.model.max_k),
        "embed_dim": int(cfg.model.embed_dim),
        "temperature": float(cfg.model.temperature),
        "cluster_iters_default": int(sum(cfg.model.cluster_iters) // 2),
        "smooth_iters_default": int(sum(cfg.model.smoothing_iters) // 2),
        "input_mean": [0.0, 0.0, 0.0],
        "input_std": [1.0, 1.0, 1.0],
        "input_scale": 1.0 / 255.0,
        "stride": 4,
        "supports_learned_refine": False,
    }

    metrics = {"score": float(score), "step": int(step)}
    if extra_metrics:
        metrics.update({k: float(v) for k, v in extra_metrics.items()})

    def _write(target: Path) -> None:
        (target / "model.npz").parent.mkdir(parents=True, exist_ok=True)
        np.savez(target / "model.npz", **weights_np)
        (target / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _write(primary)
    if mirror_dir:
        _write(Path(mirror_dir))


def export_torch_artifact(
    model_state: Dict[str, torch.Tensor],
    cfg: Config,
    score: float,
    step: int,
    out_dir: str,
    mirror_dir: Optional[str] = None,
) -> None:
    """Serialize a TorchScript model for runtime consumption.

    Args:
        model_state: state_dict() of MonolithicSegmenter.
        cfg: loaded Config object.
        score: scalar metric used for best-model selection.
        step: global step associated with the export.
        out_dir: primary export destination (e.g., models/ or run artifacts).
        mirror_dir: optional secondary destination.
    """

    patch_size = int(getattr(cfg.data, "patch_size", 256) or 256)
    patch_size = max(8, min(256, patch_size))
    example = torch.randn(1, 3, patch_size, patch_size)
    k_arg = int(max(2, min(4, cfg.model.max_k)))

    def _write(target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        model = MonolithicSegmenter(cfg.model).cpu()
        model.load_state_dict(model_state)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            traced = torch.jit.trace(lambda x: model(x, k=k_arg), example, strict=False)
        traced.save(str(target / "model.pt"))
        metrics = {"score": float(score), "step": int(step)}
        (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _write(Path(out_dir))
    if mirror_dir:
        _write(Path(mirror_dir))


def _collate_minibatch(batch: list[Dict]) -> Dict:
    view1_rgb = torch.cat([item["view1"]["rgb"] for item in batch], dim=0)
    view2_rgb = torch.cat([item["view2"]["rgb"] for item in batch], dim=0)
    grid = torch.cat([item["warp_grid"] for item in batch], dim=0)
    return {"view1_rgb": view1_rgb, "view2_rgb": view2_rgb, "grid": grid}


def smoke_export_runtime(
    out_dir: str,
    *,
    seed: int = 7,
    steps: int = 1,
    embed_dim: int = 8,
    max_k: int = 4,
    patch_size: int = 64,
    mirror_dir: Optional[str] = None,
) -> Path:
    """Produce a deterministic, CPU-friendly runtime artifact using the synthetic trainer path."""

    set_seed(seed)
    cfg = Config()
    cfg.model.embed_dim = max(1, int(embed_dim))
    cfg.model.max_k = max(2, int(max_k))
    cfg.model.cluster_iters = (1, 1)
    cfg.model.smoothing_iters = (0, 0)
    cfg.model.smoothing_lanes = ("fast",)
    cfg.model.downsample_choices = (1,)

    cfg.data.patch_size = max(8, int(patch_size))
    cfg.data.stride = cfg.data.patch_size

    cfg.train.batch_size = 1
    cfg.train.steps = max(1, int(steps))
    cfg.train.grad_accum = 1
    cfg.train.amp = False
    cfg.train.log_interval = 1

    cfg.data.max_samples = cfg.train.batch_size

    device = torch.device("cpu")
    model = MonolithicSegmenter(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    base = SyntheticDataset(num_samples=cfg.train.batch_size * 2, cfg=cfg.data)
    dataset = UnsupervisedRasterDataset(base.samples, cfg.data, cfg.aug)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=_collate_minibatch)
    it = iter(loader)

    final_loss = torch.tensor(0.0)
    for _ in range(cfg.train.steps):
        try:
            batch = next(it)
        except StopIteration:  # pragma: no cover - defensive reuse
            it = iter(loader)
            batch = next(it)

        v1_rgb = batch["view1_rgb"].to(device)
        v2_rgb = batch["view2_rgb"].to(device)
        grid = batch["grid"].to(device)

        k = min(cfg.model.max_k, 3)
        cluster_iters = cfg.model.cluster_iters[0]
        smooth_iters = cfg.model.smoothing_iters[0]

        opt.zero_grad(set_to_none=True)
        out1 = model(
            v1_rgb,
            k=k,
            downsample=1,
            cluster_iters=cluster_iters,
            smooth_iters=smooth_iters,
            smoothing_lane="fast",
        )
        out2 = model(
            v2_rgb,
            k=k,
            downsample=1,
            cluster_iters=cluster_iters,
            smooth_iters=smooth_iters,
            smoothing_lane="fast",
        )
        losses = total_loss(cfg.loss, out1["probs"], out2["probs"], v1_rgb, v2_rgb, grid)
        final_loss = losses["loss"]
        final_loss.backward()
        opt.step()

    export_torch_artifact(
        model.state_dict(),
        cfg,
        score=float(final_loss.detach().cpu()),
        step=cfg.train.steps,
        out_dir=out_dir,
        mirror_dir=mirror_dir,
    )
    return Path(out_dir)


def _main():  # pragma: no cover - exercised in docs/CLI, not default tests
    parser = argparse.ArgumentParser(description="Runtime artifact utilities")
    parser.add_argument("--smoke-out", type=str, required=True, help="Directory to write runtime artifact")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--max-k", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--mirror-out", type=str, default=None)
    args = parser.parse_args()

    smoke_export_runtime(
        out_dir=args.smoke_out,
        seed=args.seed,
        steps=args.steps,
        embed_dim=args.embed_dim,
        max_k=args.max_k,
        patch_size=args.patch_size,
        mirror_dir=args.mirror_out,
    )


if __name__ == "__main__":
    _main()


__all__ = ["export_numpy_artifacts", "export_torch_artifact", "smoke_export_runtime"]
