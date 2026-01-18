<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Training Pipeline (RGB-only)

- Data source: synthetic RGB tiles remain the default, but shard-backed ingestion is now supported via `data.source=shards` with `data.processed_root`, `data.dataset_id`, and split names (`train`, `metrics_train`, `val`). Shards follow the v0 layout under `training/datasets/processed/<dataset>/<split>/shard-xxxxx/` with `inputs/`, optional `targets/`, and `index.jsonl` describing `input`, `target`, `item_id`, and `split`.
- Dataset prep v0: offline header + shard builder stays authoritative (see [docs/dataset/HEADERS.md](../dataset/HEADERS.md)). Labeled tiles are split deterministically: 25% of labeled items go to `metrics_train` (metrics-only), the remainder to `val`; unlabeled items go to `train`.
- Loader: `training/data/dataset.py` builds two augmented RGB views per sample (flip/rotate/noise) with identity warp grids; elevation channels are not supported. A GeoTIFF path is available via `training/data/geo_patch_dataset.py` for 512x512 RGB patches with optional aligned targets (real data now primary, synthetic retained for smoke).
- Shard loader: `training/data/sharded_tif_dataset.py` streams shards as an `IterableDataset`, partitions shard directories per DataLoader worker, and supports optional per-worker LRU caching (`data.cache_mode=lru`, `data.cache_max_items`). Perf knobs: `data.num_workers` (CPU workers; 0 uses the main process), `data.prefetch_factor` (batches prefetched per worker; raise when I/O-bound), `data.persistent_workers` (keep workers alive across epochs to avoid fork cost), `data.pin_memory` (pin host buffers before GPU transfer), and cache knobs `data.cache_mode` (`none` disables caching; `lru` keeps the most recent samples) with `data.cache_max_items` (0 means unbounded within a worker).
- Metrics: Targets are used only for IoU evaluation and ignored during loss. IoU masks out labels `<=0` so unlabeled/background pixels are not penalized.
- Config: `training/config.py` defaults keep patch size 512 and random K in `[2,16]`; only Python configs are supported by default (PyYAML optional). Real-data fields `data.raster_paths`/`data.target_paths` are provided for GeoTIFF patch sampling.
- Device/AMP: choose `--device cuda|mps|cpu` (default: auto). AMP is forced on for CUDA, forced off for MPS, and only follows `--amp` on CPU.
- Smoke train (CPU-friendly):
```
python -m training.train --device cpu --steps 2 --checkpoint_dir /tmp/seg_ckpt
```
- Evaluation (synthetic proxy):
```
python -m training.eval --synthetic --seed 7
```

### Smoke runtime export (deterministic, CPU-only)
Use the synthetic trainer path to write a minimal runtime artifact for tests and offline validation:
```
python -m training.export --smoke-out /tmp/runtime_smoke --seed 7 --steps 1 --embed-dim 8 --max-k 4 --patch-size 32
```
Outputs `model.npz` + `meta.json` under the provided directory; runtime meta version is locked to the numpy path.

### Runtime adoption status
- The QGIS plugin currently runs the legacy TorchScript CNN/K-Means path. Wiring the new numpy runtime artifacts into the plugin is **deferred** until the new model is trained.

## Expectations
- Inputs are RGB tensors normalized to `[0,1]`; no DEM/elevation fields are consumed.
- Training remains separate from QGIS runtime; the plugin consumes exported numpy artifacts (`model/best`).
