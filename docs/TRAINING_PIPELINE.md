<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Training Pipeline (Manifest-backed)

- Data prep: run `scripts/data/prepare_naip_aws_3dep_dataset.py` (AWS COG) or `scripts/data/prepare_naip_3dep_dataset.py` (legacy TNM) to produce aligned RGB/DEM tiles + `manifest.jsonl` (see DATASETS.md).
- Loader: `training/data/naip_3dep_dataset.py` (aliased by `naip_aws_3dep_dataset.py`) reads the manifest, opens tiles via rasterio, normalizes RGB to [0,1], standardizes DEM per-tile, and feeds paired views through `UnsupervisedRasterDataset`.
- Config: YAML supported via PyYAML (fallback to python configs). Examples: `configs/datasets/naip_3dep_example.yaml` (legacy) and `configs/datasets/naip_aws_3dep_example.yaml` (AWS) set `data.manifest_path`, patch/stride, and smoke-friendly steps.
- Training command (smoke, CPU-friendly):
```
python -m training.train --config configs/datasets/naip_aws_3dep_example.yaml --steps 1 --amp 0 --checkpoint_dir /tmp/seg_ckpt
```
- Validation-only check:
```
python scripts/data/prepare_naip_aws_3dep_dataset.py --output-dir /tmp/naipaws3dep --validate --sample-tiles 8
```

## Expectations
- DEM is always warped to the NAIP-derived grid using -tap and matched `-tr/-te/-ts` to avoid half-pixel shifts.
- Tiles are full-coverage only (no padding); manifest paths are relative to the chosen output root.
- Training remains separate from QGIS runtime inference; TorchScript export is still out-of-scope.
