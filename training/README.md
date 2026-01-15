<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Training Overview (Unsupervised)

- Scope: placeholder contract for future unsupervised training; no training code shipped.

## Data layout (proposed)
- training/data/
  - images/: 3-channel RGB tiles or full scenes (uint8 preferred). Naming flexible; keep consistent splits.
  - eval_labels/ (optional): segmentation labels for offline evaluation only (uint8, class IDs start at 0).
  - metadata.json (optional): records source CRS/extent if needed for reprojection tests.
- Expectation: unlabeled imagery drives feature/cluster learning; labels only for metrics.

## Model export contract
- Input: tensor float32 in [0,1], shape (B, 3, H, W).
- Output TorchScript: returns tuple `(mask, features)` or `(None, features)` where:
  - mask: optional predicted labels per tile (H, W) uint8-compatible.
  - features: latent feature map (C, H', W') used by `predict_cnn` for clustering + latent KNN.
- Constraints: keep channel-first; avoid breaking `predict_cnn` expectations in [funcs.py](../funcs.py).

## Config placeholders
- Model backbone: CNN producing dense features; exact layers TBD.
- Losses: see MODEL_HISTORY.md for experiments; baseline assumes self-supervised/unsupervised objectives.
- Augmentation: color jitter, random crops/rotations recommended (no labels required).
- Hardware: PyTorch-first; GPU recommended, CPU fallback acceptable.

## Next steps
- Define concrete self-supervised loss choices and record them in MODEL_HISTORY.md.
- Add training scripts/notebooks when available; keep export contract stable.
