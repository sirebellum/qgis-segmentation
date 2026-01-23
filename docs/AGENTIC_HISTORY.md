<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# AGENTIC_HISTORY

Purpose: Phase tracking for agentic prompt iterations. Append new phases; do not rewrite prior entries.

## Phase 0 — Documentation Reconciliation (2026-01-21)
- Intent: Align documentation with actual codebase state after significant divergence.
- Summary: Rewrote all docs in `docs/` and subfolders to accurately reflect:
  - Plugin runtime uses TorchScript CNN + torch-only K-Means (no scikit-learn)
  - Global center fit + streaming assignment (no per-chunk relabeling)
  - `model/` directory (runtime_numpy, runtime_backend) does not exist
  - `scripts/datasets_ingest/` does not exist
  - `training/train.py` forwards to `train_distill.py`
  - Autoencoder reconstruction loss enabled by default
  - SLIC precomputed during shard build
- Files Touched: All docs/*.md files rewritten
- Validation: Documentation now matches CODE_DESCRIPTION.md, CODESTATE.md, and actual file structure
- Risks/Notes: Tests referencing missing packages are skip-gated

## Phase 1 — Smoothing Toggle + Discrete Sliders + Info Icons (2026-01-23)
- Intent: Add UI toggle to enable/disable post-process smoothing (default off), convert sliders to 3-state discrete controls (Low/Medium/High), add hover info icons with detailed tooltips.
- Summary:
  - Added `checkSmoothing` checkbox (default unchecked) to gate post-smoothing; renamed label from "Smoothness" to "Smoothing"
  - Converted all three sliders (`sliderSmoothness`, `sliderSpeed`, `sliderAccuracy`) from continuous (0–100) to discrete (0–2) with tick marks
  - Added `infoSmoothing`, `infoSpeed`, `infoAccuracy` QToolButton widgets with rich-text tooltips explaining each slider's effect
  - Updated `_collect_heuristics()` to use discrete slider level mapping via `SLIDER_LEVEL_VALUES`
  - Added `_is_smoothing_enabled()` helper and updated `_blur_config()` to return `None` when checkbox is unchecked
  - Updated `predict()` to conditionally log smoothing status based on checkbox state
  - UI geometry preserved; all changes are inline insertions
- Files Touched:
  - `segmenter_dialog_base.ui`: added checkbox, info icons, updated slider ranges/defaults
  - `segmenter.py`: added `SLIDER_LEVEL_*` constants, `_is_smoothing_enabled()`, updated `_collect_heuristics()`, `_blur_config()`
  - `tests/test_ui_slider_mappings.py`: new QGIS-free test file for slider mappings and defaults
  - `docs/plugin/ARCHITECTURE.md`: documented smoothing toggle and discrete sliders
  - `docs/plugin/RUNTIME_STATUS.md`: documented opt-in smoothing and UI controls
  - `docs/plugin/MODEL_NOTES.md`: documented slider effects with discrete levels
  - `docs/CODE_DESCRIPTION.md`: updated UI description
  - `docs/AGENTIC_HISTORY.md`: this entry
- Commands:
  - `python -m compileall .`
  - `python -m pytest tests/ -q`
- Validation: All tests pass; smoothing disabled by default; sliders snap to 3 states
- Risks/Notes: Old continuous slider values (if any stored in QSettings) will be clamped to nearest discrete level on read

## Phase Template
Use this format for future entries:
```
## Phase N — Title (YYYY-MM-DD)
- Intent:
- Summary:
- Files Touched:
- Commands:
- Validation:
- Risks/Notes:
```

## Phase 2 — K-Means–Only Pivot: Remove CNN Runtime (2026-01-23)
- Intent: Pivot the plugin runtime to K-Means–only segmentation. Remove CNN inference path, model dropdown, and TorchScript model loading. Re-map UI sliders to K-Means-relevant controls.
- Summary:
  - Removed model dropdown (`inputLoadModel`) from UI and all related labels
  - Repositioned Input Layer and Resolution dropdowns to fill the space
  - Updated slider tooltips to describe K-Means-specific behavior (no CNN tile sizes, no KNN neighbors)
  - Removed `execute_cnn_segmentation` from pipeline.py and funcs.py facade
  - Removed CNN imports from pipeline.py (`fit_global_cnn_centers`, `predict_cnn`, `predict_cnn_with_centers`)
  - Removed `_materialize_model` import (used for CNN model loading)
  - Removed `load_model()`, `render_models()`, `set_model()` methods from segmenter.py
  - Removed `self.model` state variable and model-related kwargs
  - Updated layer naming from `{layer}_{model}_{segments}_{resolution}` to `{layer}_kmeans_{segments}_{resolution}`
  - Added guard tests to prevent CNN code from being reintroduced:
    - `test_no_cnn_in_runtime.py`: verifies no CNN imports in runtime path
    - Updated `test_runtime_pipeline_routing.py`: checks K-Means-only exports
    - Updated `test_plugin_imports.py`: verifies K-Means-only imports
  - Updated all plugin docs (ARCHITECTURE.md, MODEL_NOTES.md, RUNTIME_STATUS.md, CODE_DESCRIPTION.md)
- Files Touched:
  - `segmenter_dialog_base.ui`: removed model dropdown, updated tooltips
  - `segmenter.py`: removed CNN path, model loading, model dropdown handlers
  - `runtime/pipeline.py`: removed CNN imports and `execute_cnn_segmentation`
  - `funcs.py`: removed CNN exports
  - `tests/test_no_cnn_in_runtime.py`: new guard test file
  - `tests/test_runtime_pipeline_routing.py`: updated for K-Means only
  - `tests/test_plugin_imports.py`: updated for K-Means only
  - `docs/plugin/ARCHITECTURE.md`: K-Means-only documentation
  - `docs/plugin/MODEL_NOTES.md`: K-Means-only documentation
  - `docs/plugin/RUNTIME_STATUS.md`: K-Means-only documentation
  - `docs/CODE_DESCRIPTION.md`: updated module descriptions
  - `docs/AGENTIC_HISTORY.md`: this entry
- Commands:
  - `python -m compileall .`
  - `python -m pytest tests/ -q`
- Validation: All tests pass; CNN path completely removed from runtime; sliders now control K-Means parameters only
- Risks/Notes:
  - `runtime/cnn.py` still exists but is not imported by the runtime segmentation path; can be deleted manually
  - `models/` directory still exists but is not used; can be deleted manually
  - Training code in `training/` is unaffected and remains isolated from plugin runtime

## Phase 3 — Apply Smoothing to K-Means + Remove Speed/Accuracy Sliders (2026-01-24)
- Intent: Wire the smoothing checkbox/slider to K-Means output; remove speed/accuracy sliders (no longer applicable); update UI layout and styling.
- Summary:
  - **Smoothing applied**: `_apply_optional_blur()` now called in `Task.finished()` after K-Means completes, gated by the smoothing checkbox
  - **UI sliders removed**: sliderSpeed, sliderAccuracy, labelSpeed, labelAccuracy, infoSpeed, infoAccuracy all removed from UI
  - **Code cleanup**: `SLIDER_LEVEL_VALUES` now only contains "smoothness"; `_collect_heuristics()` simplified accordingly
  - **predict() simplified**: removed speed/accuracy log messages; sample_scale fixed at 1.0; blur_config passed in kwargs
  - **Layout changes**:
    - Window height reduced from 520px to 400px
    - Centered alignment for Input Layer and Resolution labels
    - White text for QComboBox dropdowns (system default was too dark)
    - White background for checkbox indicator
    - White color for info icon (?) next to smoothing slider
    - Repositioned Segment/Cancel buttons to left, Feedback button to right center
  - **Tests updated**: test_ui_slider_mappings.py rewritten with 17 tests covering slider removal, styling, blur config wiring
- Files Touched:
  - `segmenter.py`: import `_apply_optional_blur`, update `Task.finished()`, simplify `SLIDER_LEVEL_VALUES`, `_collect_heuristics()`, `predict()`
  - `segmenter_dialog_base.ui`: complete rewrite removing sliders, updating layout/styling
  - `tests/test_ui_slider_mappings.py`: rewritten for new UI state
  - `docs/plugin/ARCHITECTURE.md`: removed speed/accuracy slider docs
  - `docs/plugin/RUNTIME_STATUS.md`: removed speed/accuracy slider docs, updated test count
  - `docs/CODE_DESCRIPTION.md`: updated UI description, test count
  - `docs/AGENTIC_HISTORY.md`: this entry
- Commands:
  - `python -m compileall .`
  - `python -m pytest tests/ -q` (105 passed, 4 skipped)
- Validation: All tests pass; smoothing now applies to K-Means output when checkbox enabled; speed/accuracy sliders removed from UI and code
- Risks/Notes:
  - Smoothing is optional (checkbox default off) so existing workflows are unaffected
  - Window height reduction may affect users with very low screen resolutions (420px should be safe for all modern displays)
