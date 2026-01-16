<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Agentic Required Inputs

Purpose: minimal list of files to upload when generating agentic prompt iterations; keep paths stable.

## Upload checklist (exact paths)
- docs/CODE_DESCRIPTION.md
- docs/AGENTIC_HISTORY.md
- docs/plugin/ARCHITECTURE.md
- docs/plugin/MODEL_NOTES.md
- docs/training/TRAINING_PIPELINE.md
- docs/dataset/DATASETS.md

## Regen rules
- If any checklist file is missing or moved, regenerate docs/CODE_DESCRIPTION.md to reflect new locations.
- Append-only for docs/AGENTIC_HISTORY.md; never rewrite prior phases.
- Update this file whenever doc paths or required prompt inputs change.

## Zip archive command
```bash
zip agent.zip docs/AGENTIC_REQUIRED.md docs/AGENTIC_HISTORY.md docs/CODE_DESCRIPTION.md docs/plugin/ARCHITECTURE.md docs/plugin/MODEL_NOTES.md docs/training/TRAINING_PIPELINE.md docs/dataset/DATASETS.md
```