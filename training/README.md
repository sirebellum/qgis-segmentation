<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Training Overview (Unsupervised, scaffolding)

- Purpose: next-gen monolithic unsupervised model (variable K) using RGB-only inputs; eager PyTorch only.

## Data layout
- Input tiles: RGB uint8 in `training/data/images/` (or synthetic via `--synthetic`).
- Elevation rasters are no longer consumed; dataset ingestion is being rewritten.
- No labels required; eval labels remain optional for offline checks.

## Model contract (training-time)
- Forward signature: `(rgb[B,3,512,512], K)`.
- Outputs: per-pixel probabilities `P[B,K,512,512]` (argmax are labels), latent embeddings stride/4 `E[B,D,128,128]`, prototypes, logits.
- Constraints: `2 <= K <= 16`; differentiable soft k-means/EM head with configurable iterations; two refinement lanes (fast smoothing + learned conv stub).

## Losses (implemented)
- Two-view consistency (symmetric KL on warped probabilities).
- Entropy shaping (minimize pixel entropy, maximize marginal/cluster utilization).
- Edge-aware smoothness (RGB gradients weight the penalty).

## Running
- Synthetic smoke train (CPU ok): `python -m training.train --synthetic --steps 3 --amp 0 --seed 123 --checkpoint_dir /tmp/seg_ckpt --grad-accum 2`.
- Synthetic eval: `python -m training.eval --synthetic --seed 123`.
- Config overrides: pass a python file exporting `get_config()`/`CONFIG` (see `training/config.py`).
- TensorBoard (local dashboard):
	- Training writes scalars + images to `--checkpoint_dir/tb` (or `runs/train/tb` if unset).
	- Launch with `tensorboard --logdir /tmp/seg_ckpt/tb --port 6006` and open http://localhost:6006.
	- Images include the input RGB tile and the output float probability map (no argmax/clustering applied).

### Manifest-backed ingestion
- Temporarily disabled while dataset tooling is rewritten; synthetic is the supported path.

## Knobs & randomness
- Per-batch randomization: `K ∈ {2,4,8,16}`, downsample factor {1,2}, cluster iters range, smooth iters range, smoothing lane choice.
- Elevation handling removed; RGB-only paths keep per-batch K/downsample randomization.
- Optional gradient accumulation: `--grad-accum N` (defaults to config `train.grad_accum`).

## Notes
- Training code is isolated from the QGIS runtime; no TorchScript export required in this phase.
- See [../docs/training/MODEL_HISTORY.md](../docs/training/MODEL_HISTORY.md) for experiment logging and [../docs/plugin/MODEL_NOTES.md](../docs/plugin/MODEL_NOTES.md) for the high-level contract.

## Student distillation (training-only)
- StudentCNN now emits three embeddings (coarse/mid/fine at strides 16/8/4) with coarse→mid→fine fusion and per-slice VGG-style 3×3 deep blocks. Defaults: dims 192/128/96, depths 3/2/1, GroupNorm.
- Losses per slice: feature + affinity distill against the teacher (resized per slice), soft k-means pseudo labels, edge-aware TV; optional cross-resolution consistency on assignments. Losses are merged scale-neutrally via EMA-normalized geometric mean with a separate EMA-normalized consistency term.
- Knobs: `student.*` (dims/depths/norm/dropout, enable/disable), `distill.*` (cluster iters/temperature, affinity sample, EMA decay/eps, weights). CLI overrides: `--disable-multires`, `--student-dims`, `--student-depths`, `--student-norm`, `--consistency-weight`, `--ema-decay`, `--cluster-iters`.
- Runtime: untouched; distillation remains training-only.

## Autoencoder reconstruction loss (training-only)
- Purpose: auxiliary loss encouraging blob/shape fidelity via low-pass RGB reconstruction and gradient/edge consistency at stride-4.
- Architecture: tiny training-only decoder (`TinyReconDecoder`) takes pre-projection features from student backbone and produces stride-4 RGB predictions. Decoder is excluded from deployment artifacts.
- Targets: computed from the augmented RGB input (the view the model sees), downsampled and blurred with configurable sigma.
- Loss components (Option C + D):
  - Low-pass reconstruction: L1 loss between predicted and blurred stride-4 RGB.
  - Gradient/edge consistency: L1 loss between Sobel gradient magnitudes of predicted and target RGB.
  - Combined: `L_ae = L_blur + grad_weight * L_grad`, then EMA-normalized.
- Weighting: small coefficient (`lambda_recon=0.01` default) with EMA-normalized normalization to avoid dominating other losses.
- Enabled by default: autoencoder reconstruction is now enabled by default. Use `--disable-autoencoder` to turn it off.
- Knobs: `autoencoder.*` (enabled, lambda_recon, ema_decay, blur_sigma, blur_kernel, grad_weight, detach_backbone, hidden_channels, num_blocks).
- CLI overrides: `--disable-autoencoder`, `--ae-lambda`, `--ae-ema-decay`, `--ae-blur-sigma`, `--ae-grad-weight`, `--ae-detach-backbone`.
- Saved artifacts: decoder saved separately as `decoder_training_only.pt` (clearly marked, not required for deployment).

## Training entrypoints
- **Unified entrypoint**: `python -m training.train_distill` is the primary training path.
- **Legacy passthrough**: `python -m training.train` forwards all arguments to `train_distill.py` for backwards compatibility.
