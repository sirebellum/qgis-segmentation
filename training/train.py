# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Minimal CLI trainer for synthetic or raster data."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config_loader import load_config
from .data import ShardedTifDataset, UnsupervisedRasterDataset, SyntheticDataset
from .data.geo_patch_dataset import GeoPatchViewsDataset, GeoTiffPatchDataset
from .datasets.metrics import masked_iou
from .device_policy import apply_memory_policy, dataloader_hints, select_device
from .export import export_torch_artifact
from .losses import total_loss
from .metrics import boundary_density, cluster_utilization, speckle_score, view_consistency_score
from .models.model import MonolithicSegmenter
from .utils.seed import set_seed


def _short_git_sha() -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception:
        return "nogit"


def _default_run_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{_short_git_sha()}"


def _device(preference: Optional[str]) -> torch.device:
    device = select_device(preference)
    apply_memory_policy(device)
    return device


def _colorize_labels(labels: torch.Tensor, max_k: int = 16) -> torch.Tensor:
    palette = [
        (0, 0, 0),
        (255, 99, 71),
        (135, 206, 235),
        (144, 238, 144),
        (255, 215, 0),
        (221, 160, 221),
        (255, 182, 193),
        (70, 130, 180),
        (0, 128, 128),
        (255, 140, 0),
        (152, 251, 152),
        (205, 133, 63),
        (106, 90, 205),
        (220, 20, 60),
        (95, 158, 160),
        (46, 139, 87),
        (255, 105, 180),
    ]
    palette = palette[:max_k] if max_k > 0 else palette
    cmap = torch.tensor(palette, dtype=torch.float32, device=labels.device) / 255.0
    labels_clamped = labels.clamp(min=0, max=len(cmap) - 1).long()
    colored = cmap[labels_clamped]
    return colored.permute(2, 0, 1)


def _collate(batch: list[Dict]) -> Dict:
    # Stack tensors across batch dimension.
    view1_rgb = torch.cat([item["view1"]["rgb"] for item in batch], dim=0)
    view2_rgb = torch.cat([item["view2"]["rgb"] for item in batch], dim=0)
    grid = torch.cat([item["warp_grid"] for item in batch], dim=0)
    return {
        "view1_rgb": view1_rgb,
        "view2_rgb": view2_rgb,
        "grid": grid,
    }


def _collate_eval(batch: list[Dict]) -> Dict:
    rgbs = [item["rgb"] for item in batch if item.get("rgb") is not None]
    targets = [item.get("target") for item in batch if item.get("target") is not None]
    meta = [item.get("meta") for item in batch]
    if not rgbs:
        raise ValueError("Evaluation batch is empty")
    rgb_tensor = torch.stack(rgbs, dim=0)
    target_tensor = torch.stack(targets, dim=0) if targets else None
    return {"rgb": rgb_tensor, "target": target_tensor, "meta": meta}


def _dataloader_args(cfg, device: torch.device) -> Dict:
    hints = dataloader_hints(device, cfg.data.num_workers)
    workers = hints["num_workers"]
    pin_memory = bool(cfg.data.pin_memory or device.type == "cuda")
    args: Dict = {
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(cfg.data.persistent_workers) if cfg.data.persistent_workers else hints["persistent_workers"],
    }
    if workers > 0 and cfg.data.prefetch_factor:
        args["prefetch_factor"] = int(cfg.data.prefetch_factor)
    return args


def _build_train_loader(cfg, device: torch.device) -> DataLoader:
    args = _dataloader_args(cfg, device)
    source = (cfg.data.source or "synthetic").lower()
    if source == "shards":
        if not cfg.data.dataset_id:
            raise ValueError("data.dataset_id is required when data.source='shards'")
        dataset = ShardedTifDataset(
            processed_root=cfg.data.processed_root,
            dataset_id=cfg.data.dataset_id,
            split=cfg.data.train_split,
            data_cfg=cfg.data,
            aug_cfg=cfg.aug,
            with_augmentations=True,
            include_targets=False,
            cache_mode=cfg.data.cache_mode,
            cache_max_items=cfg.data.cache_max_items,
        )
        return DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            collate_fn=_collate,
            shuffle=False,
            **args,
        )

    if source == "geo" or cfg.data.raster_paths:
        if not cfg.data.raster_paths:
            raise ValueError("raster_paths are required when data.source='geo'")
        base_ds = GeoTiffPatchDataset(
            cfg.data.raster_paths,
            targets=cfg.data.target_paths,
            data_cfg=cfg.data,
            with_targets=False,
        )
        wrapped = GeoPatchViewsDataset(base_ds, aug_cfg=cfg.aug)
        return DataLoader(
            wrapped,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            collate_fn=_collate,
            **args,
        )

    base_ds = SyntheticDataset(num_samples=cfg.train.batch_size * 2, cfg=cfg.data)
    if cfg.data.max_samples is not None:
        base_ds.samples = base_ds.samples[: cfg.data.max_samples]
    wrapped = UnsupervisedRasterDataset(base_ds.samples, cfg.data, cfg.aug)
    return DataLoader(
        wrapped,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=_collate,
        **args,
    )


def _build_eval_loader(cfg, split: str, device: torch.device) -> Optional[DataLoader]:
    source = (cfg.data.source or "synthetic").lower()
    args = _dataloader_args(cfg, device)
    if source == "shards":
        if not cfg.data.dataset_id:
            raise ValueError("data.dataset_id is required when evaluating shard splits")
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
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_eval, **args)

    if source == "geo" or cfg.data.raster_paths:
        base_ds = GeoTiffPatchDataset(
            cfg.data.raster_paths or [],
            targets=cfg.data.target_paths,
            data_cfg=cfg.data,
            with_targets=True,
        )
        return DataLoader(base_ds, batch_size=1, shuffle=False, collate_fn=_collate_eval, **args)

    return None


def _evaluate_split(model, cfg, device: torch.device, split: str) -> Optional[Dict]:
    loader = _build_eval_loader(cfg, split, device)
    if loader is None:
        return None

    total = 0
    valid = 0
    mean_values: List[float] = []
    per_class_sum: Dict[int, float] = {}
    per_class_count: Dict[int, int] = {}
    unsup: List[Dict[str, float]] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            target = batch.get("target")
            out = model(rgb, k=min(cfg.model.max_k, 4), smoothing_lane="fast")
            probs = out["probs"]
            labels = probs.argmax(dim=1).cpu().numpy()
            total += rgb.shape[0]

            if target is None:
                util = float(cluster_utilization(probs).cpu())
                speckle = float(speckle_score(torch.as_tensor(labels)).cpu())
                boundary = float(boundary_density(torch.as_tensor(labels)).cpu())
                unsup.append(
                    {
                        "utilization": util,
                        "speckle": speckle,
                        "boundary": boundary,
                        "self_consistency": float(view_consistency_score(probs, probs).cpu()),
                    }
                )
                continue

            target_np = target.cpu().numpy()
            for pred_arr, gt_arr in zip(labels, target_np):
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

    summary: Dict[str, object] = {
        "samples": total,
        "valid": valid,
        "mean_iou": mean_iou,
        "per_class": per_class_avg,
    }
    if unsup:
        # Aggregate unsupervised metrics for unlabeled splits
        summary["unsupervised_avg"] = {k: float(torch.tensor([m[k] for m in unsup]).mean()) for k in unsup[0].keys()}
    return summary


def _evaluate_synthetic(model, cfg, device: torch.device) -> Dict:
    base_ds = SyntheticDataset(num_samples=2, cfg=cfg.data)
    ds = UnsupervisedRasterDataset(base_ds.samples, cfg.data, cfg.aug)
    metrics: List[Dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for sample in ds:
            v = sample["view1"]
            rgb = v["rgb"].to(device)
            out = model(rgb, k=min(cfg.model.max_k, 4), smoothing_lane="fast")
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
    return summary


def _maybe_build_teacher(cfg, device: torch.device):
    if not cfg.teacher.enabled:
        return None
    try:
        if cfg.teacher.name == "fake":
            from .teachers.teacher_base import FakeTeacher

            teacher = FakeTeacher(embed_dim=cfg.teacher.proj_dim, device=device)
        else:
            from .teachers.dinov2 import Dinov2Teacher

            teacher = Dinov2Teacher(model_name=cfg.teacher.name, proj_dim=cfg.teacher.proj_dim, device=device)
        teacher.eval()
        return teacher
    except Exception as exc:  # pragma: no cover - teacher optional
        print(json.dumps({"event": "teacher_init_failed", "error": str(exc)}))
        return None


def main():
    parser = argparse.ArgumentParser(description="Unsupervised segmentation trainer (scaffolding)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto", help="Device preference (auto → cuda → mps → cpu)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--synthetic-data", action="store_true", help="Alias for --synthetic")
    parser.add_argument("--data-source", choices=["synthetic", "shards", "geo"], default=None, help="Override data.source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--amp", type=int, default=0, help="Enable AMP on CPU fallback; CUDA forces AMP on, MPS forces AMP off")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--run-root", type=str, default=None, help="Root directory for training runs (default: training/runs)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run identifier; defaults to timestamp+git sha")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Legacy run directory override (treated as full run path when set)")
    parser.add_argument("--logdir", type=str, default=None, help="TensorBoard log directory (default: <run>/tb)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on samples for smoke runs")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker override")
    parser.add_argument("--processed-root", type=str, default=None, help="Processed shard root (overrides data.processed_root)")
    parser.add_argument("--dataset-id", type=str, default=None, help="Dataset id for shard-backed training")
    parser.add_argument("--eval-interval", type=int, default=None, help="Optional eval interval (steps) for shard splits")
    parser.add_argument("--evaluation-epoch", type=int, default=None, help="Run evaluation every N epochs (default: 1)")
    parser.add_argument("--export-best-to", type=str, default=None, help="Directory to write best-model numpy artifacts (default: <run>/artifacts)")
    parser.add_argument("--export-training-mirror", type=str, default=None, help="Mirror directory for training-ledger copy (default: <run>/artifacts_mirror)")
    parser.add_argument("--log-image-every", type=int, default=None, help="Image logging stride (steps), default 1")
    parser.add_argument("--use-teacher", action="store_true", help="Enable teacher distillation (disabled by default)")
    parser.add_argument("--teacher-name", type=str, default=None, help="Teacher backbone name (fake|dinov2_vitl14|...)")
    parser.add_argument("--teacher-proj-dim", type=int, default=None, help="Teacher projection dim")
    parser.add_argument("--teacher-feature-weight", type=float, default=None, help="Weight for feature distillation loss")
    parser.add_argument("--teacher-affinity-weight", type=float, default=None, help="Weight for affinity distillation loss")
    parser.add_argument("--teacher-sample", type=int, default=None, help="Number of sampled points for affinity distillation")
    parser.add_argument("--no-export", action="store_true", help="Disable numpy export (useful for profiling-only runs)")
    args = parser.parse_args()

    cfg = load_config()
    if args.data_source:
        cfg.data.source = args.data_source
    elif args.synthetic or args.synthetic_data:
        cfg.data.source = "synthetic"
    if args.dataset_id:
        cfg.data.dataset_id = args.dataset_id
    if args.processed_root:
        cfg.data.processed_root = args.processed_root
    if args.eval_interval is not None:
        cfg.train.eval_interval = args.eval_interval
    if args.evaluation_epoch is not None:
        cfg.train.evaluation_epoch = max(1, args.evaluation_epoch)
    if args.steps:
        cfg.train.steps = args.steps
    if args.grad_accum:
        cfg.train.grad_accum = max(1, args.grad_accum)
    if args.max_samples is not None:
        cfg.data.max_samples = args.max_samples
    if args.num_workers is not None:
        cfg.data.num_workers = max(0, args.num_workers)
    if args.log_image_every is not None:
        cfg.train.log_image_interval = max(1, args.log_image_every)

    cfg.teacher.enabled = bool(args.use_teacher)
    if args.teacher_name:
        cfg.teacher.name = args.teacher_name
    if args.teacher_proj_dim:
        cfg.teacher.proj_dim = max(1, int(args.teacher_proj_dim))
    if args.teacher_feature_weight is not None:
        cfg.teacher.feature_weight = float(args.teacher_feature_weight)
    if args.teacher_affinity_weight is not None:
        cfg.teacher.affinity_weight = float(args.teacher_affinity_weight)
    if args.teacher_sample is not None:
        cfg.teacher.sample = max(1, int(args.teacher_sample))

    set_seed(args.seed)

    device = _device(args.device)
    if device.type == "mps":
        try:
            x = torch.ones(1, 1, 2, 2, device=device, requires_grad=True)
            grid = torch.zeros(1, 1, 1, 2, device=device)
            out = F.grid_sample(x, grid)
            out.sum().backward()
        except NotImplementedError:
            print(json.dumps({"event": "mps_fallback_cpu", "reason": "grid_sampler_backward_unsupported"}))
            device = torch.device("cpu")

    if device.type == "cuda":
        cfg.train.amp = True
    elif device.type == "mps":
        cfg.train.amp = False
    else:
        cfg.train.amp = bool(args.amp)

    model = MonolithicSegmenter(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")
    teacher = _maybe_build_teacher(cfg, device)
    teacher_losses = None
    if teacher is not None:
        from .losses_distill import affinity_distillation, feature_distillation  # type: ignore

        teacher_losses = {
            "feature": feature_distillation,
            "affinity": affinity_distillation,
        }

    source_label = (cfg.data.source or "synthetic").lower()
    if source_label == "shards":
        print(
            json.dumps(
                {
                    "event": "data_source",
                    "source": "shards",
                    "dataset_id": cfg.data.dataset_id,
                    "train_split": cfg.data.train_split,
                    "processed_root": cfg.data.processed_root,
                    "num_workers": cfg.data.num_workers,
                    "prefetch_factor": cfg.data.prefetch_factor,
                    "cache_mode": cfg.data.cache_mode,
                    "cache_max_items": cfg.data.cache_max_items,
                }
            )
        )
    else:
        print(json.dumps({"event": "data_source", "source": source_label}))

    loader = _build_train_loader(cfg, device)

    base_root = Path(args.run_root or "training/runs")
    if args.checkpoint_dir and not args.run_root:
        run_dir = Path(args.checkpoint_dir)
    else:
        run_id = args.run_id or _default_run_id()
        run_dir = base_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.logdir) if args.logdir else run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.export_best_to) if args.export_best_to else run_dir / "artifacts"
    default_mirror = Path(args.export_training_mirror) if args.export_training_mirror else Path("training/best_model")
    mirror_dir = default_mirror
    eval_root = run_dir / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(tb_dir))
    log_path = run_dir / "train_log.jsonl"
    (run_dir / "config_resolved.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    run_meta = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "git_sha": _short_git_sha(),
        "seed": args.seed,
        "device": str(device),
        "data_source": source_label,
        "num_workers": cfg.data.num_workers,
        "evaluation_epoch": cfg.train.evaluation_epoch,
        "eval_interval_steps": cfg.train.eval_interval,
        "log_image_interval": cfg.train.log_image_interval,
        "teacher_enabled": cfg.teacher.enabled,
        "teacher_name": cfg.teacher.name if cfg.teacher.enabled else None,
        "teacher_proj_dim": cfg.teacher.proj_dim if cfg.teacher.enabled else None,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    target_steps = cfg.train.steps
    steps_per_epoch = cfg.train.steps_per_epoch
    try:
        dataset_steps = len(loader)
        if dataset_steps > 0:
            steps_per_epoch = steps_per_epoch or dataset_steps
    except TypeError:
        pass
    if not steps_per_epoch or steps_per_epoch <= 0:
        steps_per_epoch = target_steps
    eval_every = max(1, cfg.train.evaluation_epoch)

    accum = max(1, cfg.train.grad_accum)
    step = 0
    micro_step = 0
    loader_iter = iter(loader)
    opt.zero_grad(set_to_none=True)
    best_score = float("inf")

    def run_eval(epoch_idx: int, global_step: int):
        epoch_dir = eval_root / f"epoch_{epoch_idx}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        summaries: Dict[str, Dict] = {}
        if source_label == "shards":
            splits = (cfg.data.metrics_split, cfg.data.val_split)
            for split_name in splits:
                try:
                    result = _evaluate_split(model, cfg, device, split_name)
                except FileNotFoundError as exc:  # pragma: no cover - dataset dependent
                    print(json.dumps({"event": "metrics_skip", "split": split_name, "reason": str(exc)}))
                    continue
                if result is None:
                    continue
                summaries[split_name] = result
                (epoch_dir / f"{split_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"eval/{split_name}/{key}", value, global_step)
                if isinstance(result.get("per_class"), dict):
                    for cls_id, value in result["per_class"].items():
                        writer.add_scalar(f"eval/{split_name}/per_class/{cls_id}", value, global_step)
                if isinstance(result.get("unsupervised_avg"), dict):
                    for key, value in result["unsupervised_avg"].items():
                        writer.add_scalar(f"eval/{split_name}/unsup/{key}", value, global_step)
        else:
            if source_label == "synthetic":
                summary = _evaluate_synthetic(model, cfg, device)
                split_name = "synthetic"
            else:
                split_name = cfg.data.val_split or "train"
                summary = _evaluate_split(model, cfg, device, split_name)
            if summary:
                summaries[split_name] = summary
                (epoch_dir / f"{split_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                for key, value in summary.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"eval/{split_name}/{key}", value, global_step)
                if isinstance(summary.get("avg"), dict):
                    for key, value in summary["avg"].items():
                        writer.add_scalar(f"eval/{split_name}/avg/{key}", value, global_step)

        if summaries:
            print(json.dumps({"event": "evaluation", "epoch": epoch_idx, "results": summaries}))
        model.train()

    while step < target_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        micro_step += 1
        knobs = model.sample_knobs()
        k = min(cfg.model.max_k, max(2, knobs["k"]))
        cluster_iters = knobs["cluster_iters"]
        smooth_iters = knobs["smooth_iters"]
        smoothing_lane = knobs["smoothing_lane"]
        downsample = knobs["downsample"]

        v1_rgb = batch["view1_rgb"].to(device)
        v2_rgb = batch["view2_rgb"].to(device)
        grid = batch["grid"].to(device)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out1 = model(
                v1_rgb,
                k=k,
                downsample=downsample,
                cluster_iters=cluster_iters,
                smooth_iters=smooth_iters,
                smoothing_lane=smoothing_lane,
            )
            out2 = model(
                v2_rgb,
                k=k,
                downsample=downsample,
                cluster_iters=cluster_iters,
                smooth_iters=smooth_iters,
                smoothing_lane=smoothing_lane,
            )
            losses = total_loss(
                cfg.loss,
                out1["probs"],
                out2["probs"],
                v1_rgb,
                v2_rgb,
                grid,
            )
            total_loss_value = losses["loss"]
            if teacher is not None and teacher_losses is not None:
                t_out = teacher.extract(v1_rgb)
                t_feat = t_out.features.to(device, non_blocking=True)
                student_feat = out1.get("embeddings")
                if student_feat is not None:
                    if student_feat.shape[-2:] != t_feat.shape[-2:]:
                        t_feat = F.interpolate(t_feat, size=student_feat.shape[-2:], mode="bilinear", align_corners=False)
                    feat_loss = teacher_losses["feature"](student_feat, t_feat)
                    aff_loss = teacher_losses["affinity"](student_feat, t_feat, sample=cfg.teacher.sample)
                    losses["teacher_feature"] = feat_loss
                    losses["teacher_affinity"] = aff_loss
                    total_loss_value = total_loss_value + cfg.teacher.feature_weight * feat_loss + cfg.teacher.affinity_weight * aff_loss
            loss = total_loss_value / float(accum)
        scaler.scale(loss).backward()

        if micro_step % accum == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1

        if step > 0 and (step % cfg.train.log_interval == 0 or step == 1):
            msg = {
                "step": step,
                "micro_step": micro_step,
                "loss": float((loss * accum).detach().cpu()),
                "consistency": float(losses["consistency"].cpu()),
                "entropy_pixel": float(losses["entropy_pixel"].cpu()),
                "entropy_marginal": float(losses["entropy_marginal"].cpu()),
                "smoothness": float(losses["smoothness"].cpu()),
                "k": k,
                "cluster_iters": cluster_iters,
                "smooth_iters": smooth_iters,
                "smoothing_lane": smoothing_lane,
                "downsample": downsample,
                "grad_accum": accum,
            }
            if "teacher_feature" in losses:
                msg["teacher_feature"] = float(losses["teacher_feature"].detach().cpu())
            if "teacher_affinity" in losses:
                msg["teacher_affinity"] = float(losses["teacher_affinity"].detach().cpu())
            print(json.dumps(msg))
            with log_path.open("a") as f:
                f.write(json.dumps(msg) + "\n")

            writer.add_scalar("loss/total", loss.detach(), step)
            writer.add_scalar("loss/consistency", losses["consistency"], step)
            writer.add_scalar("loss/entropy_pixel", losses["entropy_pixel"], step)
            writer.add_scalar("loss/entropy_marginal", losses["entropy_marginal"], step)
            writer.add_scalar("loss/smoothness", losses["smoothness"], step)
            if "teacher_feature" in losses:
                writer.add_scalar("loss/teacher_feature", losses["teacher_feature"], step)
            if "teacher_affinity" in losses:
                writer.add_scalar("loss/teacher_affinity", losses["teacher_affinity"], step)
            writer.add_scalar("hparams/k", k, step)
            writer.add_scalar("hparams/cluster_iters", cluster_iters, step)
            writer.add_scalar("hparams/smooth_iters", smooth_iters, step)
            writer.add_scalar("hparams/downsample", downsample, step)
            writer.add_scalar("hparams/grad_accum", accum, step)

            if not args.no_export:
                current_score = float(total_loss_value.detach().cpu())
                if current_score < best_score:
                    best_score = current_score
                    try:
                        export_torch_artifact(
                            model.state_dict(),
                            cfg,
                            score=current_score,
                            step=step,
                            out_dir=str(artifacts_dir),
                            mirror_dir=str(mirror_dir) if mirror_dir else None,
                        )
                        print(json.dumps({"event": "export_best", "score": current_score, "step": step}))
                    except Exception as exc:  # pragma: no cover - best-effort export
                        print(json.dumps({"event": "export_failed", "error": str(exc)}))

            if step % cfg.train.log_image_interval == 0:
                rgb_vis = v1_rgb[0].detach().cpu().clamp(0, 1)
                prob_sample = out1["probs"][0].detach().cpu()
                if prob_sample.shape[0] < 3:
                    pad = torch.zeros(3 - prob_sample.shape[0], *prob_sample.shape[1:])
                    prob_vis = torch.cat([prob_sample, pad], dim=0)
                else:
                    prob_vis = prob_sample[:3]
                labels_color = _colorize_labels(prob_sample.argmax(dim=0), max_k=cfg.model.max_k)
                writer.add_image("input/rgb", rgb_vis, step)
                writer.add_image("output/prob_float_map", prob_vis, step)
                writer.add_image("output/labels_color", labels_color, step)

        if step > 0 and step % steps_per_epoch == 0:
            epoch_idx = step // steps_per_epoch
            if epoch_idx % eval_every == 0:
                run_eval(epoch_idx, step)

    # Final evaluation if the last epoch was not aligned with the cadence
    final_epoch = math.ceil(target_steps / max(1, steps_per_epoch))
    if target_steps % steps_per_epoch != 0 or final_epoch % eval_every != 0:
        run_eval(final_epoch, target_steps)

    ckpt_path = checkpoints_dir / "model.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
