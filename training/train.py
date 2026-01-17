# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Minimal CLI trainer for synthetic or raster data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config_loader import load_config
from .data import ShardedTifDataset, UnsupervisedRasterDataset, SyntheticDataset
from .datasets.metrics import masked_iou
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


def _build_train_loader(cfg) -> DataLoader:
    args = _dataloader_args(cfg)
    if (cfg.data.source or "synthetic").lower() == "shards":
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


def _build_eval_loader(cfg, split: str) -> Optional[DataLoader]:
    if (cfg.data.source or "synthetic").lower() != "shards":
        return None
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
    args = _dataloader_args(cfg)
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_eval, **args)


def _evaluate_split(model, cfg, device: torch.device, split: str) -> Optional[Dict]:
    loader = _build_eval_loader(cfg, split)
    if loader is None:
        return None

    total = 0
    valid = 0
    mean_values: List[float] = []
    per_class_sum: Dict[int, float] = {}
    per_class_count: Dict[int, int] = {}

    model.eval()
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

    return {
        "samples": total,
        "valid": valid,
        "mean_iou": mean_iou,
        "per_class": per_class_avg,
    }


def main():
    parser = argparse.ArgumentParser(description="Unsupervised segmentation trainer (scaffolding)")
    parser.add_argument("--config", type=str, default=None, help="Config python file path")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--data-source", choices=["synthetic", "shards"], default=None, help="Override data.source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None, help="TensorBoard log directory (default: checkpoint_dir/tb or runs/train)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on samples for smoke runs")
    parser.add_argument("--processed-root", type=str, default=None, help="Processed shard root (overrides data.processed_root)")
    parser.add_argument("--dataset-id", type=str, default=None, help="Dataset id for shard-backed training")
    parser.add_argument("--eval-interval", type=int, default=None, help="Optional eval interval (steps) for shard splits")
    parser.add_argument("--export-best-to", type=str, default="model/best", help="Directory to write best-model numpy artifacts")
    parser.add_argument("--export-training-mirror", type=str, default="training/best_model", help="Mirror directory for training-ledger copy")
    parser.add_argument("--no-export", action="store_true", help="Disable numpy export (useful for profiling-only runs)")
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
    if args.eval_interval is not None:
        cfg.train.eval_interval = args.eval_interval
    if args.steps:
        cfg.train.steps = args.steps
    if args.grad_accum:
        cfg.train.grad_accum = max(1, args.grad_accum)
    cfg.train.amp = bool(args.amp)
    if args.max_samples is not None:
        cfg.data.max_samples = args.max_samples
    set_seed(args.seed)

    device = _device()
    model = MonolithicSegmenter(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

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
        print(json.dumps({"event": "data_source", "source": "synthetic"}))

    loader = _build_train_loader(cfg)

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

        if source_label == "shards" and cfg.train.eval_interval and cfg.train.eval_interval > 0:
            if step > 0 and step % cfg.train.eval_interval == 0:
                for split_name in (cfg.data.metrics_split, cfg.data.val_split):
                    try:
                        result = _evaluate_split(model, cfg, device, split_name)
                    except FileNotFoundError as exc:  # pragma: no cover - depends on dataset presence
                        print(json.dumps({"event": "metrics_skip", "split": split_name, "reason": str(exc)}))
                        continue
                    if result is None:
                        continue
                    payload = {"event": "metrics", "split": split_name}
                    payload.update(result)
                    print(json.dumps(payload))
                model.train()

    if source_label == "shards":
        for split_name in (cfg.data.metrics_split, cfg.data.val_split):
            try:
                result = _evaluate_split(model, cfg, device, split_name)
            except FileNotFoundError as exc:  # pragma: no cover - depends on dataset presence
                print(json.dumps({"event": "metrics_skip_final", "split": split_name, "reason": str(exc)}))
                continue
            if result is None:
                continue
            payload = {"event": "metrics_final", "split": split_name}
            payload.update(result)
            print(json.dumps(payload))
        model.train()

    ckpt_path = run_root / "model.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
