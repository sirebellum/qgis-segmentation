<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Training Overview (Unsupervised, scaffolding)

- Purpose: next-gen monolithic unsupervised model (variable K) with optional elevation; eager PyTorch only.

## Data layout
- Input tiles: RGB uint8 in `training/data/images/` (or synthetic via `--synthetic`).
- Optional elevation rasters matching RGB extent; resampled internally to RGB grid. Mixed batches with/without elevation are allowed.
- No labels required; eval labels remain optional for offline checks.

## Model contract (training-time)
- Forward signature: `(rgb[B,3,512,512], K, elev[B,1,512,512]|None, elev_present)`.
- Outputs: per-pixel probabilities `P[B,K,512,512]` (argmax are labels), latent embeddings stride/4 `E[B,D,128,128]`, prototypes, logits.
- Constraints: `2 <= K <= 16`; elevation injected post-encoder via gated FiLM; differentiable soft k-means/EM head with configurable iterations; two refinement lanes (fast smoothing + learned conv stub).

## Losses (implemented)
- Two-view consistency (symmetric KL on warped probabilities).
- Entropy shaping (minimize pixel entropy, maximize marginal/cluster utilization).
- Edge-aware smoothness (RGB gradients + optional elevation gradients weight the penalty).

## Running
- Synthetic smoke train (CPU ok): `python -m training.train --synthetic --steps 3 --amp 0 --seed 123 --checkpoint_dir /tmp/seg_ckpt`.
- Synthetic eval: `python -m training.eval --synthetic --seed 123`.
- Config overrides: pass a python file exporting `get_config()`/`CONFIG` (see `training/config.py`).
- TensorBoard (local dashboard):
	- Training writes scalars + images to `--checkpoint_dir/tb` (or `runs/train/tb` if unset).
	- Launch with `tensorboard --logdir /tmp/seg_ckpt/tb --port 6006` and open http://localhost:6006.
	- Images include the input RGB tile and the output float probability map (no argmax/clustering applied).

## Knobs & randomness
- Per-batch randomization: `K ∈ {2,4,8,16}`, downsample factor {1,2}, cluster iters range, smooth iters range, smoothing lane choice.
- Elevation dropout applies even when elevation is present.

## Notes
- Training code is isolated from the QGIS runtime; no TorchScript export required in this phase.
- See `training/MODEL_HISTORY.md` for experiment logging and `MODEL_NOTES.md` for the high-level contract.
