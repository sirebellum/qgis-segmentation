# SPDX-License-Identifier: BSD-3-Clause
"""Teacher→student distillation trainer (training-only; runtime untouched).

- Teacher optional and off by default (enable with --use-teacher).
- Supports shard-backed GeoTIFF patches with precomputed SLIC labels (required).
- Losses: feature/affinity distill (optional), clustering shaping, SLIC boundary priors,
  edge-aware TV, and scale-neutral EMA normalization across patch sizes.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

from .config import Config, default_config
from .data.sharded_tif_dataset import ShardedTifDataset
from .data.synthetic import SyntheticDataset
from .losses_distill import (
    affinity_distillation,
    boundary_prior_losses,
    clustering_losses,
    edge_aware_tv,
    feature_distillation,
    geometric_mean_normalized,
    normalize_with_ema,
)
from .models.student_cnn import StudentEmbeddingNet, batched_kmeans
from .teachers.teacher_base import FakeTeacher, TeacherBase
from .utils.seed import set_seed


def _parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected at least one integer value")
    return [int(p) for p in parts]


def _device_student() -> torch.device:
    if torch.cuda.device_count() > 1:
        return torch.device("cuda:1")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    mps = getattr(torch.backends, "mps", None)
    if mps and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_teacher(name: str, proj_dim: int) -> TeacherBase:
    if name == "fake":
        return FakeTeacher(embed_dim=proj_dim)
    # Lazy import to avoid pulling heavy deps when teacher is disabled.
    from .teachers.dinov2 import Dinov2Teacher  # type: ignore

    return Dinov2Teacher(model_name=name, proj_dim=proj_dim)


def _resolve_patch_sizes(cfg: Config, override: Optional[Sequence[int]] = None) -> List[int]:
    sizes = list(override) if override else list(getattr(cfg.data, "patch_sizes", []) or [])
    if not sizes:
        sizes = [cfg.data.patch_size]
    sizes = [int(s) for s in sizes if int(s) > 0]
    sizes = sorted(set(sizes))
    if not sizes:
        raise ValueError("At least one positive patch size is required")
    return sizes


def _make_data_cfg(cfg: Config, patch_size: int) -> Config:
    data_cfg = replace(cfg.data, patch_size=int(patch_size), patch_sizes=(int(patch_size),))
    cfg_copy = replace(cfg, data=data_cfg)
    return cfg_copy


def _sample_patch_size(patch_sizes: Sequence[int], policy: str) -> int:
    policy = (policy or "uniform").lower()
    if policy == "uniform":
        return random.choice(list(patch_sizes))
    raise ValueError(f"Unsupported patch_size_sampling policy: {policy}")


def _collate_shard(batch):
    rgbs = torch.stack([item["rgb"] for item in batch], dim=0)
    slic = torch.stack([item["slic"] for item in batch], dim=0)
    targets = [item.get("target") for item in batch]
    return {"rgb": rgbs, "slic": slic, "target": targets, "meta": [item.get("meta") for item in batch]}


def _collate_synth(batch):
    rgbs = torch.stack([item["rgb"] for item in batch], dim=0)
    slic = torch.zeros((rgbs.shape[0], rgbs.shape[-2], rgbs.shape[-1]), dtype=torch.int64)
    return {"rgb": rgbs, "slic": slic, "target": [None] * rgbs.shape[0], "meta": [{} for _ in range(rgbs.shape[0])]}


def _build_loader(cfg: Config, patch_size: int, device: torch.device, *, split: str = "train", include_targets: bool = False) -> DataLoader:
    cfg_local = _make_data_cfg(cfg, patch_size)
    pin_memory = device.type == "cuda"
    if (cfg_local.data.source or "").lower() == "shards":
        if not cfg_local.data.dataset_id:
            raise ValueError("dataset_id is required when data.source=shards")
        dataset = ShardedTifDataset(
            processed_root=cfg_local.data.processed_root,
            dataset_id=cfg_local.data.dataset_id,
            split=split,
            data_cfg=cfg_local.data,
            aug_cfg=cfg_local.aug,
            with_augmentations=False,
            include_targets=include_targets,
            cache_mode=cfg_local.data.cache_mode,
            cache_max_items=cfg_local.data.cache_max_items,
        )
        collate = _collate_shard
    else:
        synth = SyntheticDataset(num_samples=cfg_local.train.batch_size * 2, cfg=cfg_local.data)
        dataset = synth
        collate = _collate_synth

    shuffle = not isinstance(dataset, IterableDataset)

    return DataLoader(
        dataset,
        batch_size=cfg_local.train.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=cfg_local.data.num_workers,
        pin_memory=pin_memory,
    )


def _next_batch(loader, iterator):
    try:
        batch = next(iterator)
        return batch, iterator
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
        return batch, iterator


def _teacher_patch_size(student_patch: int, mode: str) -> int:
    mode = (mode or "match").strip().lower()
    if mode == "match":
        return student_patch
    mapping = {"high": 1024, "medium": 512, "low": 256}
    if mode not in mapping:
        raise ValueError(f"Unsupported teacher patch mode: {mode}")
    return mapping[mode]


def _boundary_mask(labels: torch.Tensor) -> torch.Tensor:
    bx = labels[:, :, 1:] != labels[:, :, :-1]
    by = labels[:, 1:, :] != labels[:, :-1, :]
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :, 1:] |= bx
    mask[:, 1:, :] |= by
    return mask


def _boundary_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    pb = _boundary_mask(pred)
    tb = _boundary_mask(target)
    inter = (pb & tb).sum().float()
    union = (pb | tb).sum().float().clamp(min=1.0)
    return float((inter / union).item())


def main():
    parser = argparse.ArgumentParser(description="Teacher-student distillation trainer")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--teacher", type=str, default="dinov2_vitl14", help="Teacher name (fake | dinov2_vitl14 | ...)")
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--use-teacher", action="store_true", help="Enable teacher losses (default off)")
    parser.add_argument("--teacher-patch-mode", type=str, default=None, choices=["match", "high", "medium", "low"], help="Resize policy for teacher input")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--logdir", type=str, default="runs/distill")
    parser.add_argument("--data-source", type=str, default=None, choices=["shards", "synthetic"], help="Training data source")
    parser.add_argument("--processed-root", type=str, default=None, help="Processed shard root")
    parser.add_argument("--dataset-id", type=str, default=None, help="Dataset id for shard loader")
    parser.add_argument("--train-split", type=str, default=None, help="Train split name (default train)")
    parser.add_argument("--val-split", type=str, default=None, help="Validation split name (default val)")
    parser.add_argument("--student-embed-dim", type=int, default=None, help="Override student embedding dim (default 96)")
    parser.add_argument("--student-depth", type=int, default=None, help="Override VGG block depth (default 3)")
    parser.add_argument("--student-norm", type=str, default=None, choices=["group", "batch", "instance", "none"], help="Normalization for deep 3x3 stacks")
    parser.add_argument("--student-dropout", type=float, default=None, help="Dropout for student conv blocks")
    parser.add_argument("--patch-sizes", type=str, default=None, help="Comma-separated patch sizes (e.g., 256,512,1024)")
    parser.add_argument("--patch-size-sampling", type=str, default=None, choices=["uniform"], help="Patch-size sampling policy (default uniform)")
    parser.add_argument("--multi-scale-mode", type=str, default=None, choices=["sample_one_scale_per_step", "per_step_all_scales"], help="Per-step scale handling")
    parser.add_argument("--ema-decay", type=float, default=None, help="Override EMA decay for scale-neutral merge")
    parser.add_argument("--cluster-iters", type=int, default=None, help="Override clustering iters for pseudo labels")
    parser.add_argument("--lambda-boundary", type=float, default=None, help="Weight for boundary encouragement across SLIC edges")
    parser.add_argument("--lambda-antimerge", type=float, default=None, help="Weight for edge-strength anti-merge term")
    parser.add_argument("--lambda-within", type=float, default=None, help="Weight for within-superpixel smoothness")
    args = parser.parse_args()

    cfg = default_config() if args.config is None else default_config()  # future: load overrides
    cfg.train.steps = args.steps
    cfg.train.batch_size = args.batch_size
    if args.data_source:
        cfg.data.source = args.data_source
    if args.processed_root:
        cfg.data.processed_root = args.processed_root
    if args.dataset_id:
        cfg.data.dataset_id = args.dataset_id
    if args.train_split:
        cfg.data.train_split = args.train_split
    if args.val_split:
        cfg.data.val_split = args.val_split
    cfg.teacher.enabled = bool(args.use_teacher)
    cfg.teacher.name = args.teacher
    if args.teacher_patch_mode:
        cfg.teacher.patch_mode = args.teacher_patch_mode
    patch_sizes_override = _parse_int_list(args.patch_sizes)
    if patch_sizes_override:
        cfg.data.patch_sizes = tuple(patch_sizes_override)
        cfg.data.patch_size = int(patch_sizes_override[0])
    if args.patch_size_sampling:
        cfg.data.patch_size_sampling = args.patch_size_sampling
    if args.multi_scale_mode:
        cfg.train.multi_scale_mode = args.multi_scale_mode
    if args.student_embed_dim:
        cfg.student.embed_dim = max(1, int(args.student_embed_dim))
    if args.student_depth:
        cfg.student.depth = max(1, int(args.student_depth))
    if args.student_norm:
        cfg.student.norm = args.student_norm
    if args.student_dropout is not None:
        cfg.student.dropout = max(0.0, float(args.student_dropout))
    if args.ema_decay is not None:
        cfg.distill.ema_decay = args.ema_decay
    if args.cluster_iters is not None:
        cfg.distill.cluster_iters = args.cluster_iters
    if args.lambda_boundary is not None:
        cfg.distill.boundary_weight = args.lambda_boundary
    if args.lambda_antimerge is not None:
        cfg.distill.antimerge_weight = args.lambda_antimerge
    if args.lambda_within is not None:
        cfg.distill.within_weight = args.lambda_within
    set_seed(cfg.train.seed)

    teacher: Optional[TeacherBase] = None
    if cfg.teacher.enabled:
        teacher = _build_teacher(cfg.teacher.name, proj_dim=args.proj_dim)
        teacher.eval()
    student_device = _device_student()
    student = StudentEmbeddingNet(
        embed_dim=cfg.student.embed_dim,
        depth=cfg.student.depth,
        norm=cfg.student.norm,
        groups=cfg.student.groups,
        dropout=cfg.student.dropout,
    ).to(student_device)

    patch_sizes = _resolve_patch_sizes(cfg, patch_sizes_override)
    loaders = {size: _build_loader(cfg, size, student_device, split=cfg.data.train_split, include_targets=False) for size in patch_sizes}
    iters = {size: iter(loader) for size, loader in loaders.items()}

    val_loader: Optional[DataLoader] = None
    if cfg.data.val_split:
        try:
            val_loader = _build_loader(cfg, patch_sizes[0], student_device, split=cfg.data.val_split, include_targets=True)
        except Exception:
            val_loader = None

    opt = torch.optim.Adam(student.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(student_device.type == "cuda"))

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / "log.jsonl"

    ema_state: Dict[str, torch.Tensor] = {}
    last_norm: Dict[str, torch.Tensor] = {}
    step = 0
    while step < cfg.train.steps:
        active_scales = patch_sizes if cfg.train.multi_scale_mode == "per_step_all_scales" else [
            _sample_patch_size(patch_sizes, cfg.data.patch_size_sampling)
        ]
        scale_logs: Dict[int, Dict[str, torch.Tensor]] = {}
        norm_losses: List[torch.Tensor] = []

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            for size in active_scales:
                loader = loaders[size]
                iterator = iters[size]
                batch, iterator = _next_batch(loader, iterator)
                iters[size] = iterator
                rgb = batch["rgb"].to(student_device)
                slic = batch["slic"].to(student_device)

                emb = student(rgb)
                fd = torch.tensor(0.0, device=student_device)
                ad = torch.tensor(0.0, device=student_device)
                if teacher is not None:
                    teacher_size = _teacher_patch_size(size, cfg.teacher.patch_mode)
                    teacher_in = rgb if teacher_size == rgb.shape[-1] else F.interpolate(
                        rgb, size=(teacher_size, teacher_size), mode="bilinear", align_corners=False
                    )
                    with torch.no_grad():
                        t_out = teacher.extract(teacher_in.to(teacher.device, non_blocking=True))
                        t_feat = t_out.features.to(student_device, non_blocking=True)
                    target_feat = F.interpolate(t_feat, size=emb.shape[-2:], mode="bilinear", align_corners=False)
                    fd = cfg.teacher.feature_weight * feature_distillation(emb, target_feat)
                    ad = cfg.teacher.affinity_weight * affinity_distillation(emb, target_feat, sample=cfg.teacher.sample)
                k_choice = random.choice(list(cfg.knobs.k_choices))
                cl, extras = clustering_losses(
                    emb,
                    k=k_choice,
                    iters=cfg.distill.cluster_iters,
                    temperature=cfg.distill.temperature,
                )
                cl = cfg.distill.clustering_weight * cl
                assign = extras.get("assign")
                if assign is None:
                    raise ValueError("clustering_losses must return assignments")
                rgb_ds = F.interpolate(rgb, size=assign.shape[-2:], mode="bilinear", align_corners=False)
                tv = cfg.distill.tv_weight * edge_aware_tv(assign, rgb_ds, weight=2.0)
                priors = boundary_prior_losses(
                    emb,
                    slic,
                    rgb,
                    lambda_boundary=cfg.distill.boundary_weight,
                    lambda_antimerge=cfg.distill.antimerge_weight,
                    lambda_within=cfg.distill.within_weight,
                )
                raw_total = fd + ad + cl + tv + priors["boundary"] + priors["antimerge"] + priors["smooth_within"]

                scale_key = f"patch_{size}"
                norm_total, ema_state = normalize_with_ema(
                    raw_total, ema_state, scale_key, cfg.distill.ema_decay, cfg.distill.merge_eps
                )
                last_norm[scale_key] = norm_total.detach()
                norm_losses.append(norm_total)
                scale_logs[size] = {
                    "loss_raw": raw_total.detach(),
                    "loss_norm": norm_total.detach(),
                    "fd": fd.detach(),
                    "ad": ad.detach(),
                    "cl": cl.detach(),
                    "tv": tv.detach(),
                    "boundary": priors["boundary"].detach(),
                    "antimerge": priors["antimerge"].detach(),
                    "within": priors["smooth_within"].detach(),
                }

            loss = sum(norm_losses) / float(max(1, len(norm_losses)))

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        virtual_combined = geometric_mean_normalized(last_norm, eps=cfg.distill.merge_eps)
        msg = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "patch_sizes": active_scales,
            "virtual_combined": float(virtual_combined.detach().cpu()),
        }
        for size, parts in scale_logs.items():
            msg[f"patch{size}_loss_raw"] = float(parts["loss_raw"].cpu())
            msg[f"patch{size}_loss_norm"] = float(parts["loss_norm"].cpu())
            msg[f"patch{size}_fd"] = float(parts["fd"].cpu())
            msg[f"patch{size}_ad"] = float(parts["ad"].cpu())
            msg[f"patch{size}_cl"] = float(parts["cl"].cpu())
            msg[f"patch{size}_tv"] = float(parts["tv"].cpu())
            msg[f"patch{size}_boundary"] = float(parts["boundary"].cpu())
            msg[f"patch{size}_antimerge"] = float(parts["antimerge"].cpu())
            msg[f"patch{size}_within"] = float(parts["within"].cpu())
        with log_path.open("a") as f:
            f.write(json.dumps(msg) + "\n")
        if step % 10 == 0:
            print(json.dumps(msg))

    if val_loader is not None:
        metrics = []
        student.eval()
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(student_device)
                slic = batch["slic"].to(student_device)
                emb = student(rgb)
                assign, _ = batched_kmeans(emb, k=cfg.model.max_k, iters=cfg.distill.cluster_iters, temperature=cfg.distill.temperature)
                pred = assign.argmax(dim=1)
                for idx, tgt in enumerate(batch.get("target", [])):
                    if tgt is None:
                        continue
                    tgt_t = tgt.to(student_device)
                    pred_up = F.interpolate(pred[idx: idx + 1].float().unsqueeze(1), size=tgt_t.shape[-2:], mode="nearest").long().squeeze(1)
                    metrics.append(_boundary_iou(pred_up, tgt_t))
        student.train()
        if metrics:
            summary = {"val_boundary_iou_mean": float(sum(metrics) / len(metrics))}
            with (logdir / "val_metrics.json").open("w") as handle:
                json.dump(summary, handle)

    torch.save({"student": student.state_dict()}, logdir / "student.pt")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
