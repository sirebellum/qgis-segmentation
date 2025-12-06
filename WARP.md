# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a QGIS plugin that uses machine learning (PyTorch CNN or K-Means clustering) to segment maps into discrete regions (e.g., trees, roads, buildings, water). The plugin is designed to run within QGIS's Python environment and uses a custom dependency manager to bootstrap required packages.

## Architecture

### Core Components

**Plugin Entry Point**: `segmenter.py`
- Main plugin class (`Segmenter`) implementing the QGIS plugin interface
- Handles UI initialization, toolbar/menu integration, and task orchestration
- Uses QGIS's `QgsTask` system for asynchronous processing
- Manages status updates through `StatusEmitter` signal system

**Segmentation Algorithms**: `funcs.py`
- `predict_kmeans()`: Fast clustering-based segmentation on downsampled blocks
- `predict_cnn()`: Deep learning-based segmentation using pre-trained TorchScript models
- `tile_raster()`: Handles spatial tiling with padding for both algorithms

**Dependency Management**: `dependency_manager.py`
- Auto-installs torch, numpy, and scikit-learn into `vendor/` subdirectory
- Isolated from QGIS Python environment to avoid conflicts
- Respects environment variables: `SEGMENTER_SKIP_AUTO_INSTALL`, `SEGMENTER_TORCH_INDEX_URL`, `SEGMENTER_TORCH_SPEC`, `SEGMENTER_PYTHON`
- On macOS: defaults to CPU-only PyTorch
- On Linux/Windows: defaults to CUDA 12.1 build

**QGIS Integration**: `qgis_funcs.py`
- `render_raster()`: Converts numpy arrays to GeoTIFF and adds as QGIS layer

**Pre-trained Models**: `models/`
- TorchScript models: `model_4.pth`, `model_8.pth`, `model_16.pth` (corresponding to high/medium/low resolution)
- Models are loaded via `torch.jit.load()` and produce 384-channel feature vectors

### Data Flow

1. User selects raster layer, model type (K-Means/CNN), resolution (high/medium/low), and number of segments
2. `predict()` method reads raster via GDAL, sets up args/kwargs based on model type
3. Task is queued via `run_task()` which creates a `QgsTask` for background processing
4. For CNN: raster is tiled → CNN extracts features → K-Means clusters features → upsampled to original resolution
5. For K-Means: raster is downsampled → K-Means directly clusters RGB values → upsampled
6. `render_raster()` creates temporary GeoTIFF in system temp directory and adds to QGIS project

## Development Commands

### Testing
```bash
pytest func_test.py
```
Tests cover `predict_kmeans()`, `predict_cnn()`, and `tile_raster()` with various array shapes and parameters.

### Plugin Packaging
Uses `pb_tool` (QGIS Plugin Builder tool):
```bash
pb_tool deploy
```
This packages the plugin according to `pb_tool.cfg` and deploys to QGIS plugin directory.

### Manual Dependency Installation
If auto-install fails or for offline machines, open QGIS Python console:
```python
import sys
import subprocess
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "torch==2.2.2",
    "scikit-learn>=1.1,<2.0",
    "numpy>=1.23,<2.0",
    "--target",
    "/path/to/segmenter/vendor",
])
```

## Key Technical Details

### Resolution Mapping
- "high" → 4 pixels per block
- "medium" → 8 pixels per block  
- "low" → 16 pixels per block

Lower resolution = less noise but less detail. The resolution parameter controls the downsampling factor for K-Means and determines which CNN model is loaded.

### CNN Model Loading
Models must be loaded using `torch.jit.load()` from BytesIO objects. They expect:
- Input: `(batch, 3, tile_size, tile_size)` normalized to [0, 1]
- Output: `[reconstructed_image, feature_vectors]` where feature_vectors shape is `(batch, 384, height//4, width//4)`

### Status Callbacks
All segmentation functions accept a `status_callback` parameter. Use `_emit_status(callback, message)` helper to safely emit progress updates that appear in the plugin dialog's log panel.

### QGIS Environment Constraints
- Cannot use standard print/console output (use `QgsMessageLog.logMessage()` or status callbacks)
- Must use QGIS's task system for long-running operations to avoid UI freezing
- Raster layers must be converted to numpy arrays via GDAL
- Coordinate reference system (CRS) must be preserved when rendering output

## Plugin Metadata
- Name: Map Segmenter
- Minimum QGIS Version: 3.0
- Current Version: 2.2.0
- Repository: https://github.com/sirebellum/qgis-segmentation

## Dependencies
- PyTorch 2.2.2 (CPU on macOS, CUDA 12.1 elsewhere)
- scikit-learn >=1.1,<2.0
- numpy >=1.23,<2.0
- GDAL (provided by QGIS)
- QGIS PyQt5 bindings (provided by QGIS)
