<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# MODEL_HISTORY

Purpose: Record architectural choices, losses, and evaluation signals for training experiments.

## Current Architecture

### MonolithicSegmenter
- Stride-4 encoder with residual blocks
- Soft k-means head with differentiable assignments
- Fast smoothing lane (depthwise box filter) as default
- Learned refiner available but not used in runtime exports

### StudentEmbeddingNet
- Single stride-4 embedding path (input-size invariant)
- Two stride-2 stems → VGG-style 3×3 blocks → projection head
- GroupNorm default, configurable embed_dim/depth

### TinyReconDecoder (training-only)
- 2-block decoder taking pre-projection features
- Produces stride-4 RGB predictions
- Excluded from deployment artifacts

## Loss Components

### Unsupervised
- Two-view consistency (symmetric KL on warped probabilities)
- Entropy shaping (minimize pixel entropy, maximize marginal)
- Edge-aware smoothness (RGB gradients weight penalty)

### Distillation
- Feature distillation (cosine similarity to teacher)
- Affinity distillation (sampled pixel-pair similarities)
- Clustering shaping (soft k-means assignments)
- SLIC boundary priors (boundary/antimerge/within losses)
- Edge-aware TV on assignment maps

### Reconstruction (enabled by default)
- Low-pass reconstruction: L1 on Gaussian-blurred stride-4 RGB
- Gradient consistency: L1 on Sobel magnitude
- Combined with EMA normalization, small weight (0.01)

## Key Decisions

### Global Centers (Plugin Runtime)
- K-Means and CNN both fit centers once per run
- Streaming assignment across chunks
- No per-chunk relabeling → consistent labels

### Patch-Size Training
- Single stride-4 student shared across patch sizes
- Default schedule: 256/512/1024
- Per-scale EMA normalization + geometric mean metric

### SLIC Precompute
- Superpixels computed during shard build (not at training time)
- Required by loader (`data.require_slic=true`)
- Fallback to grid if OpenCV contrib unavailable

## Experiment Log Template
```
### YYYY-MM-DD — Experiment Name
- Config: key settings
- Loss mix: weights used
- Data: source, augmentations
- Hardware: device, batch size
- Results: metrics observed
- Notes: observations, regressions
```

## Future Work
- Wire student artifacts into plugin runtime
- Evaluate learned refiner vs fast smoothing
- Multi-scale consistency losses
