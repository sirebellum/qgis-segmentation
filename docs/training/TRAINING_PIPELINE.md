<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Training Pipeline (RGB-only)

- Data source (current state): synthetic RGB tiles only. Manifest-backed ingestion is temporarily disabled while dataset tooling is rewritten.
- Loader: `training/data/dataset.py` builds two augmented RGB views per sample (flip/rotate/noise) with identity warp grids; elevation channels are not supported.
- Config: `training/config.py` defaults keep patch size 512 and random K in `[2,16]`; only Python configs are supported by default (PyYAML optional).
- Smoke train (CPU-friendly):
```
python -m training.train --steps 2 --amp 0 --checkpoint_dir /tmp/seg_ckpt
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
