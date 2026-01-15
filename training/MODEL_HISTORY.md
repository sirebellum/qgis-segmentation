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
