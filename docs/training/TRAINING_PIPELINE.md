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

## Expectations
- Inputs are RGB tensors normalized to `[0,1]`; no DEM/elevation fields are consumed.
- Training remains separate from QGIS runtime; the plugin consumes exported numpy artifacts (`model/best`).
