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

### 2026-01-17 — Teacher→Student distillation scaffold
- Config: GeoTIFF patch loader (RGB-only, 512x512) with optional targets; teacher frozen (default DINOv2 or fake fallback); student CNN embeddings (stride 4, embed_dim configurable, param budget <10M).
- Loss mix: feature distillation (cosine), affinity distillation (sampled similarities), clustering shaping (k-means-style assignments with balance/sharpness), edge-aware TV. Random K handled via clustering head; teacher never exported.
- Hardware: teacher prefers cuda:0, student/backward on cuda:1 when available; falls back gracefully to single GPU/CPU.
- Runtime: unchanged; no teacher or student shipped yet. Exports remain numpy-only for legacy path; distillation artifacts are for training checkpoints.

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

### 2026-01-17 — Shard ingestion + metrics-only labels
- Config: `data.source=shards` with `processed_root`, `dataset_id`, worker/prefetch/cache knobs; synthetic remains default.
- Data: v0 processed shards (`train` unlabeled; `metrics_train` + `val` labeled). Targets used only for IoU metrics; labels `<=0` are ignored during scoring.
- Loop: train/eval build shard DataLoaders; periodic/final IoU computed on metrics/val splits; loss path unchanged (unsupervised two-view).
- Notes: plugin runtime still legacy TorchScript CNN/K-Means; runtime adoption of new numpy artifacts remains deferred.

### 2026-01-18 — Multires student distillation
- Architecture: StudentCNN now emits three embedding slices (coarse/mid/fine, strides 16/8/4) with channel dims 192/128/96. Coarse→mid→fine fusion uses upsample + concat + 1×1 fuse followed by VGG-like 3×3 stacks per slice (stride-preserving, GroupNorm default).
- Losses: per-slice feature + affinity distill (teacher resized per slice), soft k-means pseudo labels, edge-aware TV; optional cross-resolution consistency (coarse→mid, mid→fine) on assignment maps. Losses merged scale-neutrally via EMA-normalized geometric mean; consistency normalized separately.
- Knobs: `student.*` (dims/depths/norm/dropout, enable/disable), `distill.*` (cluster iters/temperature, affinity sample, EMA decay/eps, weights). CLI overrides: `--disable-multires`, `--student-dims`, `--student-depths`, `--student-norm`, `--consistency-weight`, `--ema-decay`, `--cluster-iters`.
- Runtime: unchanged; training-only update. Tests added for multires shapes and scale-neutral merge invariants.

### 2026-01-19 — Patch-size single-scale distillation
- Architecture: StudentCNN simplified to a single stride-4 embedding path (input-size invariant) shared across patch sizes. Default patch schedule samples 256/512/1024 inputs; no fused multi-slice heads remain.
- Losses: per-patch-size feature + affinity distill (teacher resized to stride-4 grid), soft k-means pseudo labels, edge-aware TV. Each patch size maintains its own EMA normalizer; the per-step loss optimizes the normalized total for the sampled size. A virtual geometric-mean metric combines normalized per-scale losses to stay scale-neutral.
- Knobs: `data.patch_sizes`, `data.patch_size_sampling`, and `train.multi_scale_mode` (sample one size per step vs sum all sizes) plus `student.*` (embed_dim/depth/norm/dropout) and `distill.*` (cluster iters, EMA decay/eps, loss weights). CLI keeps `--patch-sizes`, `--patch-size-sampling`, `--multi-scale-mode`, `--ema-decay`, and student overrides.
- Runtime: unchanged; distillation remains training-only. Tests updated to assert stride-4 outputs for 256/512/1024 inputs and scale-neutral EMA/geometric-mean merge behavior.

### 2026-01-21 — Autoencoder reconstruction loss (training-only)
- Architecture: added training-only `TinyReconDecoder` that takes pre-projection stride-4 features from the student backbone and produces stride-4 RGB predictions. Decoder is 2 blocks of 3×3 conv + GroupNorm + ReLU + 1×1 to 3 channels + sigmoid.
- Losses: Option C (low-pass reconstruction) + Option D (gradient/edge consistency). Targets computed from augmented RGB input: downsample to stride-4, apply Gaussian blur (sigma configurable), compute Sobel gradient magnitude. Combined loss: `L_ae = L_blur + grad_weight * L_grad`, then EMA-normalized and scaled by small coefficient (default 0.01).
- Knobs: `autoencoder.*` (enabled, lambda_recon, ema_decay, blur_sigma, blur_kernel, grad_weight, detach_backbone, hidden_channels, num_blocks). CLI overrides: `--enable-autoencoder`, `--ae-lambda`, `--ae-ema-decay`, `--ae-blur-sigma`, `--ae-grad-weight`, `--ae-detach-backbone`.
- Runtime: unchanged; decoder is training-only and excluded from student.pt artifact. Decoder saved separately as `decoder_training_only.pt` (marked as training-only, not required for deployment).
- Tests: 27 new tests covering target correctness, Sobel gradients, decoder shape, EMA normalization, training-step integration, and training-only separation.
