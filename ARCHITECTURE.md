<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Architecture

- Scope: QGIS 3 plugin "Map Segmenter"; code is source of truth.

## End-to-end flow
- User opens dialog from plugin action in [segmenter.py](segmenter.py); UI defined by [segmenter_dialog_base.ui](segmenter_dialog_base.ui) and loaded via [segmenter_dialog.py](segmenter_dialog.py).
- `Segmenter.predict()` validates 3-band GDAL raster selection, parses segments/resolution/heuristics, and dispatches a `QgsTask` via `run_task()`.
- Task runs `legacy_kmeans_segmentation()` or `legacy_cnn_segmentation()` (both in [funcs.py](funcs.py)) with cancellation token + progress callbacks.
- Segmentation outputs are blurred (optional) then rendered to a temporary GeoTIFF with source extent/CRS through [qgis_funcs.py](qgis_funcs.py) `render_raster()` and added to the QGIS project.

## Module responsibilities
- [segmenter.py](segmenter.py): plugin entrypoint; menu/toolbar wiring; dialog lifecycle; progress bar/status logging; model selection; heuristic overrides; `QgsTask` wrapper; TorchScript loader from `models/model_<resolution>.pth`; triggers adaptive profiling.
- [segmenter_dialog.py](segmenter_dialog.py) / [segmenter_dialog_base.ui](segmenter_dialog_base.ui): Qt dialog shell and widgets (layer/model/resolution selectors, sliders, progress/log, buttons).
- [funcs.py](funcs.py): numerical engine. Raster/materialization, K-Means path (`predict_kmeans`), CNN tiling/batching/prefetch (`predict_cnn`), latent KNN refinement, blur, and cancellation helpers.
- [qgis_funcs.py](qgis_funcs.py): writes numpy labels to temp GeoTIFF and registers a `QgsRasterLayer` with opacity preserved.
- [dependency_manager.py](dependency_manager.py): on-demand vendor install of torch/numpy/scikit-learn into `vendor/`, honoring env toggles (`SEGMENTER_*`), retries CPU wheels on CUDA failure.
- [perf_tuner.py](perf_tuner.py): profiles device throughput once to select `AdaptiveSettings` (safety factor/prefetch); caches JSON at `perf_profile.json`; integrates with `predict_cnn`.
- [raster_utils.py](raster_utils.py): `ensure_channel_first` utility for GDAL writes.
- [models/](models): TorchScript CNN weights (`model_4.pth`, `model_8.pth`, `model_16.pth`) consumed by `Segmenter.load_model()`.
- [metadata.txt](metadata.txt): QGIS plugin metadata (name, version 2.2.1, author, tracker/homepage).

## Key extension points / config
- Model choice: K-Means vs CNN (`Segmenter.model`, dropdown in dialog).
- Resolution presets: `RESOLUTION_CHOICES` in [segmenter.py](segmenter.py) map labels to block sizes; speed slider scales effective resolution; heuristics tweak tile size, blur kernel, latent KNN, sampling scale.
- Blur: `_legacy_blur_config` controls kernel/iterations; applied post-segmentation.
- Env toggles: `SEGMENTER_SKIP_AUTO_INSTALL`, `SEGMENTER_TORCH_SPEC`, `SEGMENTER_TORCH_INDEX_URL`, `SEGMENTER_PYTHON`, `SEGMENTER_SKIP_PROFILING` govern dependency/profiling behavior.
- Profiling cache: `perf_profile.json` in plugin dir; `load_or_profile_settings()` chooses tiered `AdaptiveSettings` and options based on device.
- Rendering: `render_raster()` uses tempdir GeoTIFF; layer name includes input layer, model, segments, resolution.

## Runtime/perf notes
- GPU/CPU selection: prefers CUDA, then MPS, else CPU in [segmenter.py](segmenter.py); `funcs._coerce_torch_device` ensures safe device objects.
- Batch/tiling: `_recommended_batch_size` respects `memory_budget` and prefetch depth; `_prefetch_batches` uses threads or CUDA streams.
- Cancellation: `CancellationToken` checked via `_maybe_raise_cancel` across loops; task cancel cancels token and updates UI.
- Logging/progress: worker emits status strings parsed by `_maybe_update_progress_from_message` to keep UI progress bar moving; history buffered in dialog log.

## Training scaffolding (isolated)
- Location: [training/](training) (pure PyTorch, eager mode). Not wired into QGIS runtime.
- Components: config dataclasses, synthetic/optional raster datasets with paired-view augmentations, monolithic model (encoder stride/4, elevation FiLM injection, soft k-means head, fast/learned refinement lanes), unsupervised losses (consistency, entropy shaping, edge-aware smoothness), proxy metrics, CLI train/eval runners.
- Contract: forward(rgb, K, elev?) → probabilities [B,K,512,512] (K ≤ 16) + embeddings stride/4; elevation optional and gated.
- CLI: `python -m training.train --synthetic --steps 3` for smoke training; `python -m training.eval --synthetic` for proxy metrics.
