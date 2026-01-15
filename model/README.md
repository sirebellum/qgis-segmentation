<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# model/

Canonical runtime artifact location for the next-gen variable-K segmenter (numpy path).

## Layout
- `best/` — latest exported model for plugin/runtime consumption.
  - `model.npz` — numpy weights (encoder + seed projection).
  - `meta.json` — metadata (version, max_k, temperature, smoothing defaults, normalization).
  - `metrics.json` — optional score ledger for the exported checkpoint.

## Producer
`python -m training.train` exports the best checkpoint automatically (unless `--no-export` is set) to both:
- `model/best/` (runtime pickup)
- `training/best_model/` (training ledger copy)

## Consumer
`segmenter.py` loads `model/best` when the "Next-Gen (Numpy)" model is selected. No torch import is required at runtime.
