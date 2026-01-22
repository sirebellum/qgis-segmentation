<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Training Baseline (Jan 2026)

## Scope
Current RGB-only PyTorch training scaffold with distillation, shard ingestion, and numpy export. Plugin runtime uses legacy TorchScript path; new artifacts not yet consumed.

## Verified Sources
- [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md): detailed pipeline docs
- [training/README.md](../../training/README.md): quick start and contract
- [training/config.py](../../training/config.py): dataclass defaults
- [training/train_distill.py](../../training/train_distill.py): unified trainer

## Data Paths

### Synthetic (default)
- `training/data/synthetic.py`: random RGB tensors
- `training/data/dataset.py`: two-view augmentation wrapper
- Safe for offline/CI testing

### Shard-backed
- `data.source=shards` enables GeoTIFF shard loading
- Shards under `training/datasets/processed/<dataset>/<split>/shard-xxxxx/`
- `index.jsonl` per shard with `input`, optional `target`, `item_id`, `split`
- SLIC labels precomputed during shard build

### GeoTIFF Patches
- `training/data/geo_patch_dataset.py`: random windows
- `data.raster_paths` / `data.target_paths` for paths
- `data.patch_sizes` for size schedule (256/512/1024)

## Training Entrypoint
```bash
# Unified (forwards to train_distill.py)
python -m training.train --synthetic --steps 3 --seed 123

# Direct distillation
python -m training.train_distill --synthetic --steps 3 --seed 123
```

## Key Features

### Autoencoder Reconstruction (enabled by default)
- `autoencoder.enabled=true` in config
- TinyReconDecoder produces stride-4 RGB
- Low-pass + gradient consistency losses
- Decoder saved separately (`decoder_training_only.pt`)
- Excluded from deployment

### Student Model
- Single stride-4 embedding path
- Input-size invariant across patch sizes
- Per-scale EMA normalization

### SLIC Boundary Priors
- Superpixels precomputed during shard build
- `data.require_slic=true` by default
- Boundary/antimerge/within losses

## Export (training/export.py)
Produces numpy artifacts:
- `model.npz`: renamed weights
- `meta.json`: version, max_k, embed_dim, temperature, stride=4
- `metrics.json`: score, step

**Note**: Plugin does not yet consume these artifacts.

## Config Highlights
```python
# Model
model.embed_dim = 96
model.max_k = 16

# Data
data.patch_sizes = (256, 512, 1024)
data.source = "synthetic"
data.require_slic = True

# Autoencoder (on by default)
autoencoder.enabled = True
autoencoder.lambda_recon = 0.01

# Student
student.embed_dim = 96
student.depth = 3
student.norm = "group"
```

## Tests (Offline)
```bash
.venv/bin/python -m pytest training/tests/ -q
```

**82 tests passing** covering:
- **Models**: test_model_shapes.py, test_model_forward_contract.py
- **Student**: test_student_embed.py (3 tests)
- **Losses**: test_losses.py, test_distill_losses_smoke.py, test_multires_losses.py
- **Autoencoder**: test_autoencoder_losses.py (27 tests)
- **Boundary priors**: test_boundary_priors.py, test_loss_slic_priors.py
- **Datasets**: test_sharded_dataset_*.py, test_geo_patch_dataset.py
- **Augmentations**: test_augmentations.py (3 tests)
- **Smoke train**: test_smoke_train.py
- **Misc**: test_synthetic.py, test_teacher_fallback.py, test_knobs_sampling.py, test_metrics_*.py

## Known Gaps
- Plugin uses TorchScript, not exported numpy artifacts
- DINOv2 teacher optional; FakeTeacher fallback for offline
