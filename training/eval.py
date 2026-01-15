# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Lightweight evaluation runner for proxy metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config_loader import load_config
from .data.dataset import UnsupervisedRasterDataset
from .data.synthetic import SyntheticDataset
from .metrics import boundary_density, cluster_utilization, speckle_score, view_consistency_score
from .models.model import MonolithicSegmenter
from .utils.seed import set_seed


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Evaluation for unsupervised scaffolding")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = _device()
    model = MonolithicSegmenter(cfg.model).to(device)
    model.eval()

    if args.synthetic or True:
        base_ds = SyntheticDataset(num_samples=2, with_elevation=cfg.data.allow_mixed_elevation)
    else:
        raise NotImplementedError("Real data eval not implemented")
    ds = UnsupervisedRasterDataset(base_ds.samples, cfg.data, cfg.aug)

    metrics = []
    for sample in ds:
        v = sample["view1"]
        rgb = v["rgb"].to(device)
        elev = v.get("elev")
        elev = elev.to(device) if elev is not None else None
        out = model(
            rgb,
            k=min(cfg.model.max_k, 4),
            elev=elev,
            elev_present=elev is not None,
            smoothing_lane="fast",
        )
        probs = out["probs"]
        labels = probs.argmax(dim=1)
        metrics.append(
            {
                "utilization": float(cluster_utilization(probs).cpu()),
                "speckle": float(speckle_score(labels).cpu()),
                "boundary": float(boundary_density(labels).cpu()),
                "self_consistency": float(view_consistency_score(probs, probs).cpu()),
            }
        )

    summary = {
        "count": len(metrics),
        "avg": {k: float(torch.tensor([m[k] for m in metrics]).mean()) for k in metrics[0].keys()},
    }
    print(json.dumps(summary))
    Path("eval_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
