# SPDX-License-Identifier: BSD-3-Clause
"""Teacher→student distillation trainer (training-only; runtime untouched).

- Teacher on GPU0 (or CPU if unavailable), student on GPU1 when present.
- Supports GeoTIFF patch dataset (RGB-only) and synthetic fallback.
- Losses: feature distill, affinity distill, clustering shaping, edge-aware TV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader

from .config import Config, default_config
from .data.dataset import UnsupervisedRasterDataset
from .data.geo_patch_dataset import GeoTiffPatchDataset
from .data.synthetic import SyntheticDataset
from .losses_distill import affinity_distillation, clustering_losses, edge_aware_tv, feature_distillation
from .models.student_cnn import StudentEmbeddingNet
from .teachers.dinov2 import Dinov2Teacher
from .teachers.teacher_base import FakeTeacher, TeacherBase
from .utils.seed import set_seed


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
    return Dinov2Teacher(model_name=name, proj_dim=proj_dim)


def _geo_dataset(cfg: Config) -> Optional[GeoTiffPatchDataset]:
    if not cfg.data.raster_paths:
        return None
    return GeoTiffPatchDataset(cfg.data.raster_paths, targets=cfg.data.target_paths, data_cfg=cfg.data, with_targets=bool(cfg.data.target_paths))


def _synthetic_dataset(cfg: Config):
    base = SyntheticDataset(num_samples=cfg.train.batch_size * 2, cfg=cfg.data)
    return UnsupervisedRasterDataset(base.samples, cfg.data, cfg.aug)


def _collate_geo(batch):
    rgbs = torch.stack([item.rgb for item in batch], dim=0)
    targets = [item.target for item in batch]
    return {"rgb": rgbs, "target": targets, "meta": [item.meta for item in batch]}


def main():
    parser = argparse.ArgumentParser(description="Teacher-student distillation trainer")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--teacher", type=str, default="fake", help="Teacher name (fake | dinov2_vitl14 | ...)")
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--logdir", type=str, default="runs/distill")
    args = parser.parse_args()

    cfg = default_config() if args.config is None else default_config()  # future: load overrides
    cfg.train.steps = args.steps
    cfg.train.batch_size = args.batch_size
    set_seed(cfg.train.seed)

    teacher = _build_teacher(args.teacher, proj_dim=args.proj_dim)
    student_device = _device_student()
    student = StudentEmbeddingNet(embed_dim=args.proj_dim).to(student_device)

    dataset = _geo_dataset(cfg) or _synthetic_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=_collate_geo if cfg.data.raster_paths else None)

    opt = torch.optim.Adam(student.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(student_device.type == "cuda"))

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / "log.jsonl"

    step = 0
    loader_iter = iter(loader)
    while step < cfg.train.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        rgb = batch["rgb"].to(student_device)
        with torch.no_grad():
            t_out = teacher.extract(rgb.to(teacher.device, non_blocking=True))
            t_feat = t_out.features.to(student_device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            s_emb = student(rgb)
            fd = feature_distillation(s_emb, t_feat)
            ad = affinity_distillation(s_emb, t_feat, sample=256)
            cl, extras = clustering_losses(s_emb.detach(), k=4, iters=3, temperature=0.8)
            tv = edge_aware_tv(extras["assign"], rgb, weight=2.0)
            loss = fd + ad + cl + 0.1 * tv
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        msg = {"step": step, "loss": float(loss.detach().cpu()), "fd": float(fd.detach().cpu()), "ad": float(ad.detach().cpu()), "cl": float(cl.detach().cpu()), "tv": float(tv.detach().cpu())}
        with log_path.open("a") as f:
            f.write(json.dumps(msg) + "\n")
        if step % 10 == 0:
            print(json.dumps(msg))

    torch.save({"student": student.state_dict()}, logdir / "student.pt")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
