# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Export helpers for numpy-only runtime artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .config import Config


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
        "version": "phase1-numpy-v0",
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


__all__ = ["export_numpy_artifacts"]
