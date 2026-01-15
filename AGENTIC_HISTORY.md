<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# AGENTIC_HISTORY

## Phase 0 — Baseline (2026-01-15)
- Intent: Establish Phase 0 baseline and generate internal docs of record from repo inspection.
- Summary:
  - Created ARCHITECTURE.md and MODEL_NOTES.md from verified code structure (segmenter UI, funcs engine, dependency/perf tooling, rendering path).
  - Added AGENTIC_HISTORY.md to track agent work; noted license duality (existing BSD-3-Clause, new docs carry MIT SPDX per request).
- Files Touched:
  - Added: ARCHITECTURE.md, MODEL_NOTES.md, AGENTIC_HISTORY.md.
  - Unmodified: existing source, LICENSE retained as BSD-3-Clause.
- Commands:
  - pytest ./func_test.py (pre-existing run, exit code 130; no new commands executed in this phase).
- Validation:
  - Confirmed absence of prior ARCHITECTURE/MODEL_NOTES/AGENTIC_HISTORY before creation.
  - Cross-checked module paths referenced in docs exist: segmenter.py, funcs.py, qgis_funcs.py, dependency_manager.py, perf_tuner.py, autoencoder_utils.py, raster_utils.py, models/ files.
- Risks/Notes:
  - License mismatch: repo LICENSE is BSD-3-Clause while new docs use MIT SPDX headers; reconcile licensing direction in a future phase.

## Phase 1 — Training Docs + License Align (2026-01-15)
- Intent: Align doc headers to BSD-3-Clause and add training contract/history scaffolding per unsupervised plan.
- Summary:
  - Switched ARCHITECTURE.md, MODEL_NOTES.md, AGENTIC_HISTORY.md headers to BSD-3-Clause for consistency with repo LICENSE.
  - Added training/README.md with unlabeled data layout and model export contract notes; added training/MODEL_HISTORY.md for recording losses/eval choices.
- Files Touched:
  - Modified: ARCHITECTURE.md, MODEL_NOTES.md, AGENTIC_HISTORY.md.
  - Added: training/README.md, training/MODEL_HISTORY.md.
- Commands:
  - pytest ./func_test.py (pre-existing run, exit code 0; no new commands run this phase).
- Validation:
  - Verified new training docs reference existing modules and exported contract expected by `predict_cnn`.
  - Confirmed training/ path created with BSD headers.
- Risks/Notes:
  - Unsupervised training details still TBD; MODEL_HISTORY.md is a placeholder to be populated with experiments.
