<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Training Pipeline (RGB-only)

## Entrypoints

### Unified Training
```bash
python -m training.train --synthetic --steps 3 --seed 123
```
Note: `training/train.py` forwards to `training/train_distill.py`.

### Direct Distillation
```bash
python -m training.train_distill --synthetic --steps 3 --seed 123
```

### Evaluation
```bash
python -m training.eval --synthetic --seed 7
```

## Data Sources

### Synthetic (default)
- RGB-only tensors from `training/data/synthetic.py`
- Wrapped with two-view augmentations
- Safe for offline/CI testing

### Shard-backed (`data.source=shards`)
- GeoTIFF tiles from `training/datasets/processed/<dataset>/<split>/shard-xxxxx/`
- Iterable loader with per-worker partitioning
- Optional LRU caching (`data.cache_mode=lru`)
- SLIC labels precomputed during shard build (required by default)

### GeoTIFF Patches
- Random windows from `data.raster_paths` via `training/data/geo_patch_dataset.py`
- Patch sizes: `data.patch_sizes` (default 256/512/1024)

## Augmentations
Defined in `training/augmentations.py`:
- **Geometry**: 90° rotations, H/V flips (synchronized for RGB/SLIC/targets)
- **Photometric**: noise, contrast, saturation (RGB only)
- Deterministic via `aug.seed`

## Model Architecture

### MonolithicSegmenter (training/models/model.py)
- Stride-4 encoder (`training/models/backbone.py`)
- Soft k-means head (`training/models/soft_cluster.py`)
- Optional learned refiner (`training/models/refine.py`)
- Fast smoothing lane (default)

### StudentEmbeddingNet (training/models/student_cnn.py)
- Single stride-4 embedding path
- Input-size invariant across patch sizes
- VGG-style 3×3 blocks with GroupNorm

## Losses

### Unsupervised (training/losses.py)
- Two-view consistency (symmetric KL)
- Entropy shaping (pixel min, marginal max)
- Edge-aware smoothness

### Distillation (training/losses_distill.py)
- Feature distillation (cosine)
- Affinity distillation (sampled similarities)
- Clustering shaping (k-means assignments)
- SLIC boundary priors
- Edge-aware TV

### Reconstruction (training/losses_recon.py)
**Enabled by default** (`autoencoder.enabled=true`):
- TinyReconDecoder: 2-block decoder producing stride-4 RGB
- Low-pass reconstruction: L1 on blurred RGB
- Gradient consistency: L1 on Sobel magnitude
- EMA-normalized weighting
- Decoder excluded from deployment artifacts

## Teacher (optional, off by default)
- Enable with `--use-teacher`
- DINOv2 or FakeTeacher fallback
- Training-only (never exported)

## Config Knobs

### Key Settings (training/config.py)
```python
model.embed_dim = 96
model.max_k = 16
model.temperature = 0.8

data.patch_sizes = (256, 512, 1024)
data.source = "synthetic"  # or "shards"
data.require_slic = True

autoencoder.enabled = True
autoencoder.lambda_recon = 0.01
autoencoder.blur_sigma = 1.0

student.embed_dim = 96
student.depth = 3
student.norm = "group"

distill.cluster_iters = 3
distill.ema_decay = 0.9

train.multi_scale_mode = "sample_one_scale_per_step"
```

### CLI Overrides
```bash
--synthetic               # Use synthetic data
--steps N                 # Training steps
--seed N                  # Random seed
--device cuda|mps|cpu     # Device selection
--amp 0|1                 # AMP toggle (auto for CUDA)
--disable-autoencoder     # Turn off reconstruction loss
--ae-lambda F             # Reconstruction weight
--patch-sizes 256,512     # Patch size schedule
```

## Export (training/export.py)
Exports numpy artifacts for future runtime:
- `model.npz`: renamed weights
- `meta.json`: version, max_k, embed_dim, temperature, stride
- `metrics.json`: best loss, step

**Note**: Plugin does not yet consume these artifacts.

## Device Policy
- `--device cuda|mps|cpu` (default: auto)
- AMP forced on for CUDA, off for MPS
- Teacher prefers cuda:0, student/backward on cuda:1 when available

## Metrics
- Targets used only for IoU evaluation (not in loss)
- Labels ≤0 masked during IoU computation
- Proxy metrics: utilization, speckle, boundary density
