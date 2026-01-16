<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# MODEL_HISTORY

- Purpose: record architectural choices, losses, and evaluation signals for unsupervised training. Append new entries as experiments run.

## Current baseline (placeholder)
- Architecture: TorchScript CNN emitting latent feature map; mask output optional (can be None).
- Losses (planned): contrastive or clustering-friendly objectives (e.g., InfoNCE, DeepCluster-style), plus spatial smoothness/TV regularizers.
- Evaluation signals (planned): clustering stability across crops, feature diversity metrics, optional IoU/accuracy if eval_labels exist.
- Export contract: output tuple `(mask, features)`; `features` consumed by `predict_cnn` latent KNN in [funcs.py](../funcs.py).

## Future entries
- Add dated sections noting: config, loss mix, augmentations, hardware, metrics, and any regressions/improvements.

### 2026-01-15 — Next-gen unsupervised scaffold
- Config: default in `training/config.py` (K randomized in [2,16], soft k-means head, FiLM elevation gate, stride/4 embeddings).
- Loss mix: consistency (symmetric KL), entropy min/max (pixel vs marginal), edge-aware smoothness.
- Data: synthetic paired views with optional elevation + dropout; geometry shared; photometric jitter per view.
- Metrics tracked: cluster utilization, speckle, boundary density, view consistency.
- Notes: eager PyTorch only; refinement lanes (fast box blur, learned conv stub) selectable by knob.

### 2026-01-15 — Elevation masks + grad accumulation
- Config tweaks: per-sample elevation masks honored end-to-end (collate, FiLM gate); fast smoothing kernel fixed for any K; grad accumulation exposed via CLI/config.
- Data: synthetic default path remains; real rasters still stubbed. Mixed elevation/non-elevation batches supported via masks.
- Training loop: optional `--grad-accum` averaging loss; logs include micro_step vs step.
- Notes: synthetic-first by design; real IO to be added in later phases once raster backend is wired.
