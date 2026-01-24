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
## Phase 4 — Map-to-Raster Assist for Web Service Layers (2026-01-23)
- Intent: Auto-launch Convert map to raster dialog when user selects a web service or vector layer, prefilled with canvas extent + 1 map unit/pixel, without auto-running.
- Summary:
  - **New module**: `map_to_raster.py` with pure-python layer detection helpers:
    - `is_file_backed_gdal_raster()`: detects valid 3-band GDAL GeoTIFFs
    - `is_renderable_non_file_layer()`: detects WMS/WMTS/XYZ/ArcGIS/vector layers
    - `build_convert_map_to_raster_params()`: builds algorithm parameters (extent, layer, MAP_UNITS_PER_PIXEL=1.0)
    - `open_convert_map_to_raster_dialog()`: opens processing dialog without executing
  - **segmenter.py changes**:
    - `render_layers()` now shows all renderable layers (rasters + vectors + web services), not just file-backed rasters
    - Added `_on_layer_selection_changed()` to detect non-file layers and trigger assist
    - Added `_open_convert_map_to_raster_assist()` to build params and open dialog
    - Added `_last_map_assist_layer_id` state to prevent dialog spam on repeated selection
    - Updated `predict()` to show helpful message when non-raster is selected
  - **README updated**: documented both raster input (direct) and map/web service input (assisted conversion) workflows
  - **Tests**: `tests/test_map_to_raster.py` with 25+ QGIS-free tests for detection logic and parameter mapping
- Files Touched:
  - `map_to_raster.py`: new module
  - `segmenter.py`: import new module, update `render_layers()`, add selection handler
  - `tests/test_map_to_raster.py`: new test file
  - `README.md`: added Input Types section
  - `docs/plugin/ARCHITECTURE.md`: added map_to_raster.py to Support modules, documented map-to-raster assist contract
  - `docs/plugin/RUNTIME_STATUS.md`: added Map-to-Raster Assist section
  - `docs/CODE_DESCRIPTION.md`: documented map_to_raster.py and UI assist behavior
  - `docs/AGENTIC_HISTORY.md`: this entry
- Commands:
  - `python -m compileall .`
  - `python -m pytest tests/ -q`
- Validation: All tests pass; selecting web service layer opens Convert map to raster dialog; selecting file raster proceeds normally
- Risks/Notes:
  - Dialog only opens once per layer selection (spam prevention via `_last_map_assist_layer_id`)
  - Processing dialog does NOT auto-run; user must adjust settings and click Run
  - If `native:rasterize` algorithm is unavailable, falls back to qgis:rasterize or gdal:rasterize_over_fixed_value

## Phase 4.1 — Map-to-Raster Dialog Trigger Refinement + Palette-Aware Styling (2026-01-23)
- Intent: Only show Convert map to raster dialog when map is the only layer OR user explicitly selects it; make all UI text palette-aware for dark/light mode.
- Summary:
  - **Dialog trigger logic refined**: `_on_layer_selection_changed` now only opens dialog if:
    1. The selected layer is a web service/vector (not file-backed GDAL raster), AND
    2. Either it's the only layer in the dropdown (`count() == 1`), OR user explicitly selected it (`count() > 1`)
  - **Palette-aware styling**: Updated `segmenter_dialog_base.ui` stylesheet to use `palette(text)` for all text elements:
    - Added `QToolButton { color: palette(text); }` to global stylesheet
    - Added `QProgressBar { color: palette(text); }` to global stylesheet
    - Added `QSlider { color: palette(text); }` to global stylesheet
    - Removed hardcoded `color: white` from `infoSmoothing` button (now inherits from QToolButton style)
  - **Tests updated**: Modified `test_info_smoothing_exists` to check for palette-aware styling; added `test_all_text_widgets_palette_aware` and `test_no_hardcoded_white_text_color`
- Files Touched:
  - `segmenter.py`: refined `_on_layer_selection_changed` logic
  - `segmenter_dialog_base.ui`: updated stylesheet, removed hardcoded white color
  - `tests/test_ui_slider_mappings.py`: updated and added palette-aware tests
  - `docs/AGENTIC_HISTORY.md`: this entry
- Commands:
  - `python -m compileall .`
  - `python -m pytest tests/ -q` (131 passed, 4 skipped)
- Validation: All tests pass; UI text adapts to system dark/light mode; dialog only shows when appropriate
- Risks/Notes:
  - Palette styling relies on Qt's system palette detection; should work on all platforms
  - No visual change for users already in dark mode; light mode users will now see dark text

## Phase 5 — Seam Prevention: Halo Overlap + Global Block Alignment (2026-01-23)
- Intent: Eliminate visible chunk boundary seams in K-Means segmentation by implementing halo overlap for smoothing context, globally-aligned block grid, and fixed feature scaling.
- Summary:
  - **Root cause**: Visible seams at chunk boundaries due to: (1) edge pixels lacking neighbor context during 3x3 avg_pool2d smoothing, (2) block grid aligned to chunk-local coordinates instead of global, (3) inconsistent feature representation at boundaries.
  - **Fix #1 - Halo overlap**: Added `_expand_window_for_halo()` and `_crop_halo_from_result()` utilities in `chunking.py`. Chunks are now read with 1 extra pixel on each edge (DESCRIPTOR_HALO_PIXELS=1) to provide context for the 3x3 smoothing kernel.
  - **Fix #2 - Global block alignment**: Added `_compute_globally_aligned_descriptors()` in `kmeans.py` that computes block indices from absolute raster coordinates, ensuring the low-res label grid is consistent across chunks.
  - **Fix #3 - Fixed scaling**: Verified that descriptors use float32 with no per-chunk normalization; added test to guard against per-chunk scaling.
  - **Updated `_assign_blocks_streaming()`**: Now reads chunks with halo, computes globally-aligned descriptors, and writes output to correct global block positions.
  - **Docstring added**: `kmeans.py` module docstring documents seam prevention strategies.
- Files Touched:
  - `runtime/chunking.py`: added `DEFAULT_HALO_PIXELS`, `_expand_window_for_halo()`, `_crop_halo_from_result()`, updated `__all__`
  - `runtime/kmeans.py`: added module docstring, `DESCRIPTOR_HALO_PIXELS`, `_compute_globally_aligned_descriptors()`, `_expand_window_for_halo()` (local), updated `_assign_blocks_streaming()`, updated `__all__`
  - `tests/test_seam_halo_alignment.py`: new test file with 15 tests covering halo expansion, cropping, global alignment, seam prevention, and scaling consistency
  - `docs/plugin/ARCHITECTURE.md`: documented seam prevention contract
  - `docs/plugin/RUNTIME_STATUS.md`: added Seam Prevention section, updated test count to 146
  - `docs/CODE_DESCRIPTION.md`: documented halo utilities and seam prevention in kmeans.py, updated test count
  - `docs/AGENTIC_HISTORY.md`: this entry
- Commands:
  - `python -m py_compile runtime/kmeans.py runtime/chunking.py`
  - `python -m pytest tests/ -q` (146 passed, 4 skipped)
- Validation:
  - All 146 tests pass (131 existing + 15 new seam prevention tests)
  - Single-chunk vs multi-chunk segmentation match >95% (test_single_chunk_vs_multi_chunk_consistency)
  - No per-chunk normalization detected (test_no_per_chunk_normalization)
  - Uniform image produces consistent labels across chunk boundaries (test_no_visible_seam_pattern)
- Risks/Notes:
  - Halo size is conservative (1px) and may need increase if larger smoothing kernels are added
  - Edge chunks have reduced halo at raster boundaries (handled via clamping)
  - Global alignment adds minimal overhead since block indices are computed from absolute coordinates

## Phase 5.1 — Block-Level Overlap Stitching for Seam Elimination (2026-01-23)
- Intent: Eliminate remaining 1px grid artifact at chunk boundaries by implementing block-level overlap with deterministic last-write-wins stitching.
- Root Cause Analysis:
  - Previous halo approach only provided pixel-level context (DESCRIPTOR_HALO_PIXELS)
  - The 3x3 avg_pool2d smoothing kernel and subsequent block pooling meant edge blocks at chunk boundaries saw different neighbor context than interior blocks
  - Even with pixel halo, the boundary block descriptors could differ between adjacent chunks because smoothing computation depended on chunk-local position
  - The 1px grid appeared exactly at block boundaries where chunks met
- Summary:
  - **Block overlap**: Added BLOCK_OVERLAP=1 constant; chunks now overlap by 1 block in each direction
  - **Stride calculation**: `_block_chunk_plan()` now computes stride as `chunk_size - overlap` instead of `chunk_size`
  - **Last-write-wins stitching**: All blocks from each chunk are written; later chunks overwrite earlier in overlap regions (deterministic row-major order)
  - **Pixel halo increased**: DESCRIPTOR_HALO_PIXELS raised from 1 to 3 for additional edge stability
  - **Simplified streaming**: Removed complex interior-only crop logic; simple overwrite is cleaner and equally effective
- Files Touched:
  - `runtime/kmeans.py`: added `BLOCK_OVERLAP=1`, updated `_block_chunk_plan()` stride calculation, simplified `_assign_blocks_streaming()` docstring and stitching logic, updated `__all__`
  - `runtime/chunking.py`: updated `DEFAULT_HALO_PIXELS` to 3
  - `tests/test_seam_halo_alignment.py`: added `BLOCK_OVERLAP` import, added `TestBlockOverlap` class with 3 tests (constant exists, overlap prevents grid, stride uses overlap)
  - `docs/plugin/RUNTIME_STATUS.md`: expanded Seam Prevention section with 5-point strategy, updated Key Invariants
  - `docs/plugin/ARCHITECTURE.md`: updated seam prevention contract
  - `docs/AGENTIC_HISTORY.md`: this entry
- Commands:
  - `python -m py_compile runtime/kmeans.py runtime/chunking.py`
  - `python -m pytest tests/ -q` (149 passed, 4 skipped)
- Validation:
  - All 149 tests pass (146 existing + 3 new block overlap tests)
  - Single-chunk vs multi-chunk match ratio ≥90% (test_overlap_prevents_1px_grid)
  - Chunk iteration correctly uses overlapping stride (test_chunk_iteration_uses_overlap_stride)
  - BLOCK_OVERLAP ≥ 1 enforced (test_block_overlap_constant_exists)
- Risks/Notes:
  - Block overlap slightly increases total computation (each edge block computed by 2 chunks)
  - Memory overhead minimal since only block indices overlap, not full chunk data
  - Deterministic stitching ensures reproducible results across runs
