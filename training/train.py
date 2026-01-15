# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Minimal CLI trainer for synthetic or raster data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config_loader import load_config
from .data.dataset import UnsupervisedRasterDataset
from .data.naip_3dep_dataset import build_dataset_from_manifest
from .data.synthetic import SyntheticDataset
from .losses import total_loss
from .models.model import MonolithicSegmenter
from .export import export_numpy_artifacts
from .utils.seed import set_seed


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _stack_optional(tensors: list[Optional[torch.Tensor]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    present = [t is not None for t in tensors]
    if not any(present):
        return None, None
    template = next(t for t in tensors if t is not None)
    filled = [t if p else torch.zeros_like(template) for t, p in zip(tensors, present)]
    stacked = torch.cat(filled, dim=0)
    mask = torch.tensor(present, dtype=torch.bool).view(-1, 1, 1, 1)
    return stacked, mask


def _collate(batch: list[Dict]) -> Dict:
    # Stack tensors across batch dimension.
    view1_rgb = torch.cat([item["view1"]["rgb"] for item in batch], dim=0)
    view2_rgb = torch.cat([item["view2"]["rgb"] for item in batch], dim=0)
    view1_elev, view1_mask = _stack_optional([item["view1"].get("elev") for item in batch])
    view2_elev, view2_mask = _stack_optional([item["view2"].get("elev") for item in batch])
    grid = torch.cat([item["warp_grid"] for item in batch], dim=0)
    return {
        "view1_rgb": view1_rgb,
        "view2_rgb": view2_rgb,
        "view1_elev": view1_elev,
        "view2_elev": view2_elev,
        "view1_mask": view1_mask,
        "view2_mask": view2_mask,
        "grid": grid,
    }


def build_dataloader(cfg, synthetic: bool, with_elev: bool, manifest_path: Optional[str] = None) -> DataLoader:
    if synthetic:
        base_ds = SyntheticDataset(num_samples=cfg.train.batch_size * 2, with_elevation=with_elev)
        wrapped = UnsupervisedRasterDataset(base_ds.samples, cfg.data, cfg.aug)
    else:
        if manifest_path:
            cfg.data.manifest_path = manifest_path
        base_ds = build_dataset_from_manifest(cfg.data, cfg.aug)
        wrapped = base_ds
    return DataLoader(wrapped, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=_collate)


def main():
    parser = argparse.ArgumentParser(description="Unsupervised segmentation trainer (scaffolding)")
    parser.add_argument("--config", type=str, default=None, help="Config python file path")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None, help="TensorBoard log directory (default: checkpoint_dir/tb or runs/train)")
    parser.add_argument("--manifest", type=str, default=None, help="Path to NAIP/3DEP manifest.jsonl")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on samples for smoke runs")
    parser.add_argument("--export-best-to", type=str, default="model/best", help="Directory to write best-model numpy artifacts")
    parser.add_argument("--export-training-mirror", type=str, default="training/best_model", help="Mirror directory for training-ledger copy")
    parser.add_argument("--no-export", action="store_true", help="Disable numpy export (useful for profiling-only runs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.steps:
        cfg.train.steps = args.steps
    if args.grad_accum:
        cfg.train.grad_accum = max(1, args.grad_accum)
    cfg.train.amp = bool(args.amp)
    if args.manifest:
        cfg.data.manifest_path = args.manifest
    if args.max_samples is not None:
        cfg.data.max_samples = args.max_samples
    set_seed(args.seed)

    device = _device()
    model = MonolithicSegmenter(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    use_synth = not bool(cfg.data.manifest_path)
    if cfg.data.manifest_path:
        print(f"Loading NAIP/3DEP manifest dataset: {cfg.data.manifest_path}")
    else:
        print("Using synthetic dataset (manifest not provided)")
    loader = build_dataloader(cfg, synthetic=use_synth, with_elev=cfg.data.allow_mixed_elevation, manifest_path=cfg.data.manifest_path)

    run_root = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("runs") / "train"
    run_root.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.logdir) if args.logdir else run_root / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))
    log_path = run_root / "train_log.jsonl"

    target_steps = cfg.train.steps
    accum = max(1, cfg.train.grad_accum)
    step = 0
    micro_step = 0
    loader_iter = iter(loader)
    opt.zero_grad(set_to_none=True)
    best_score = float("inf")

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
        v1_elev = batch["view1_elev"].to(device) if batch["view1_elev"] is not None else None
        v2_elev = batch["view2_elev"].to(device) if batch["view2_elev"] is not None else None
        v1_mask = batch["view1_mask"].to(device) if batch["view1_mask"] is not None else None
        v2_mask = batch["view2_mask"].to(device) if batch["view2_mask"] is not None else None

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out1 = model(
                v1_rgb,
                k=k,
                elev=v1_elev,
                elev_present=v1_mask if v1_mask is not None else v1_elev is not None,
                downsample=downsample,
                cluster_iters=cluster_iters,
                smooth_iters=smooth_iters,
                smoothing_lane=smoothing_lane,
            )
            out2 = model(
                v2_rgb,
                k=k,
                elev=v2_elev,
                elev_present=v2_mask if v2_mask is not None else v2_elev is not None,
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
                v1_elev,
                v2_elev,
            )
            loss = losses["loss"] / float(accum)
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
            print(json.dumps(msg))
            with log_path.open("a") as f:
                f.write(json.dumps(msg) + "\n")

            writer.add_scalar("loss/total", loss.detach(), step)
            writer.add_scalar("loss/consistency", losses["consistency"], step)
            writer.add_scalar("loss/entropy_pixel", losses["entropy_pixel"], step)
            writer.add_scalar("loss/entropy_marginal", losses["entropy_marginal"], step)
            writer.add_scalar("loss/smoothness", losses["smoothness"], step)
            writer.add_scalar("hparams/k", k, step)
            writer.add_scalar("hparams/cluster_iters", cluster_iters, step)
            writer.add_scalar("hparams/smooth_iters", smooth_iters, step)
            writer.add_scalar("hparams/downsample", downsample, step)
            writer.add_scalar("hparams/grad_accum", accum, step)

            if not args.no_export:
                current_score = msg["loss"]
                if current_score < best_score:
                    best_score = current_score
                    try:
                        export_numpy_artifacts(
                            model.state_dict(),
                            cfg,
                            score=current_score,
                            step=step,
                            out_dir=args.export_best_to,
                            mirror_dir=args.export_training_mirror,
                        )
                        print(json.dumps({"event": "export_best", "score": current_score, "step": step}))
                    except Exception as exc:  # pragma: no cover - best-effort export
                        print(json.dumps({"event": "export_failed", "error": str(exc)}))

            # Images: input RGB and float probability map (first 3 channels padded if needed)
            rgb_vis = v1_rgb[0].detach().cpu().clamp(0, 1)
            prob_sample = out1["probs"][0].detach().cpu()
            if prob_sample.shape[0] < 3:
                pad = torch.zeros(3 - prob_sample.shape[0], *prob_sample.shape[1:])
                prob_vis = torch.cat([prob_sample, pad], dim=0)
            else:
                prob_vis = prob_sample[:3]
            writer.add_image("input/rgb", rgb_vis, step)
            writer.add_image("output/prob_float_map", prob_vis, step)

    ckpt_path = run_root / "model.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
