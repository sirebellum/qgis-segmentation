<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Agentic Required Inputs

Purpose: minimal list of files to upload when generating agentic prompt iterations; keep paths stable.

## Upload checklist (exact paths)
- docs/CODE_DESCRIPTION.md
- docs/CODESTATE.md
- docs/AGENTIC_HISTORY.md
- docs/AGENTIC_REQUIRED.md
- docs/plugin/ARCHITECTURE.md
- docs/plugin/MODEL_NOTES.md
- docs/plugin/RUNTIME_STATUS.md
- docs/training/TRAINING_PIPELINE.md
- docs/training/MODEL_HISTORY.md
- docs/training/TRAINING_BASELINE.md
- docs/dataset/DATASETS.md
- training/README.md

## Regen rules
- If any checklist file is missing or moved, regenerate docs/CODE_DESCRIPTION.md to reflect new locations.
- Append-only for docs/AGENTIC_HISTORY.md; never rewrite prior phases.
- Update this file whenever doc paths or required prompt inputs change.

## Zip archive command
```bash
cd docs && zip agent.zip CODE_DESCRIPTION.md CODESTATE.md AGENTIC_HISTORY.md AGENTIC_REQUIRED.md plugin/ARCHITECTURE.md plugin/MODEL_NOTES.md plugin/RUNTIME_STATUS.md training/TRAINING_PIPELINE.md training/MODEL_HISTORY.md training/TRAINING_BASELINE.md dataset/DATASETS.md ../training/README.md
```