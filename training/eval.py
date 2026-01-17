# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Lightweight evaluation runner for proxy metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .config_loader import load_config
from .data import ShardedTifDataset, UnsupervisedRasterDataset, SyntheticDataset
from .datasets.metrics import masked_iou
from .metrics import boundary_density, cluster_utilization, speckle_score, view_consistency_score
from .models.model import MonolithicSegmenter
from .utils.seed import set_seed


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate_eval(batch: list[Dict]) -> Dict:
    rgbs = [item["rgb"] for item in batch if item.get("rgb") is not None]
    targets = [item.get("target") for item in batch if item.get("target") is not None]
    meta = [item.get("meta") for item in batch]
    if not rgbs:
        raise ValueError("Evaluation batch is empty")
    rgb_tensor = torch.stack(rgbs, dim=0)
    target_tensor = torch.stack(targets, dim=0) if targets else None
    return {"rgb": rgb_tensor, "target": target_tensor, "meta": meta}


def _dataloader_args(cfg) -> Dict:
    workers = max(0, int(cfg.data.num_workers))
    args: Dict = {
        "num_workers": workers,
        "pin_memory": bool(cfg.data.pin_memory),
        "persistent_workers": bool(cfg.data.persistent_workers and workers > 0),
    }
    if workers > 0 and cfg.data.prefetch_factor:
        args["prefetch_factor"] = int(cfg.data.prefetch_factor)
    return args


def main():
    parser = argparse.ArgumentParser(description="Evaluation for unsupervised scaffolding")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--data-source", choices=["synthetic", "shards"], default=None)
    parser.add_argument("--dataset-id", type=str, default=None, help="Dataset id for shard-backed evaluation")
    parser.add_argument("--processed-root", type=str, default=None, help="Processed shard root override")
    parser.add_argument("--split", type=str, default=None, help="Split to evaluate (default: data.val_split)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_source:
        cfg.data.source = args.data_source
    elif args.synthetic:
        cfg.data.source = "synthetic"
    if args.dataset_id:
        cfg.data.dataset_id = args.dataset_id
    if args.processed_root:
        cfg.data.processed_root = args.processed_root
    set_seed(args.seed)
    device = _device()
    model = MonolithicSegmenter(cfg.model).to(device)
    model.eval()

    source_label = (cfg.data.source or "synthetic").lower()

    if source_label == "shards":
        split = args.split or cfg.data.val_split
        if not cfg.data.dataset_id:
            raise ValueError("data.dataset_id is required when data.source='shards'")
        dataset = ShardedTifDataset(
            processed_root=cfg.data.processed_root,
            dataset_id=cfg.data.dataset_id,
            split=split,
            data_cfg=cfg.data,
            aug_cfg=cfg.aug,
            with_augmentations=False,
            include_targets=True,
            cache_mode=cfg.data.cache_mode,
            cache_max_items=cfg.data.cache_max_items,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_eval, **_dataloader_args(cfg))

        total = 0
        valid = 0
        mean_values: List[float] = []
        per_class_sum: Dict[int, float] = {}
        per_class_count: Dict[int, int] = {}

        with torch.no_grad():
            for batch in loader:
                rgb = batch["rgb"].to(device)
                target = batch.get("target")
                if target is None:
                    total += rgb.shape[0]
                    continue
                target_np = target.cpu().numpy()
                out = model(rgb, k=min(cfg.model.max_k, 4), smoothing_lane="fast")
                labels = out["probs"].argmax(dim=1).cpu().numpy()
                for pred_arr, gt_arr in zip(labels, target_np):
                    total += 1
                    metrics = masked_iou(pred_arr, gt_arr, ignore_label_leq=cfg.data.iou_ignore_label_leq)
                    if not metrics.get("has_valid_labels"):
                        continue
                    valid += 1
                    mean = metrics.get("mean_iou")
                    if mean is not None:
                        mean_values.append(float(mean))
                    for cls_id, value in metrics.get("per_class", {}).items():
                        cls_int = int(cls_id)
                        per_class_sum[cls_int] = per_class_sum.get(cls_int, 0.0) + float(value)
                        per_class_count[cls_int] = per_class_count.get(cls_int, 0) + 1

        per_class_avg = {
            cls: per_class_sum[cls] / max(1, per_class_count.get(cls, 0)) for cls in per_class_sum
        }
        mean_iou = float(sum(mean_values) / len(mean_values)) if mean_values else None
        summary = {
            "source": "shards",
            "split": split,
            "samples": total,
            "valid": valid,
            "mean_iou": mean_iou,
            "per_class": per_class_avg,
        }
    else:
        base_ds = SyntheticDataset(num_samples=2, cfg=cfg.data)
        ds = UnsupervisedRasterDataset(base_ds.samples, cfg.data, cfg.aug)

        metrics = []
        for sample in ds:
            v = sample["view1"]
            rgb = v["rgb"].to(device)
            out = model(
                rgb,
                k=min(cfg.model.max_k, 4),
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
            "source": "synthetic",
            "count": len(metrics),
            "avg": {k: float(torch.tensor([m[k] for m in metrics]).mean()) for k in metrics[0].keys()},
        }

    print(json.dumps(summary))
    Path("eval_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
