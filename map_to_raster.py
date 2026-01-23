# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Utilities for detecting non-file raster layers and launching Convert map to raster dialog.

This module provides pure-python helpers for layer type detection plus QGIS integration
for opening the processing dialog with prefilled parameters.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

# Constants for the processing algorithm
CONVERT_MAP_TO_RASTER_ALG_ID = "native:rasterize"
MAP_UNITS_PER_PIXEL_DEFAULT = 1.0

# Provider names that indicate web/tile services (not local GDAL files)
WEB_SERVICE_PROVIDERS = frozenset({
    "wms",
    "wmts",
    "wcs",
    "arcgismapserver",
    "arcgisfeatureserver",
    "wfs",
    "xyz",
    "vectortile",
    "mbtiles",  # MBTiles can be local but is tile-based
})

# File extensions supported for direct segmentation (GDAL-backed rasters)
SUPPORTED_RASTER_EXTENSIONS = frozenset({".tif", ".tiff"})


def is_file_backed_gdal_raster(
    provider_name: Optional[str],
    source_path: Optional[str],
    band_count: int,
) -> bool:
    """Check if a layer is a file-backed GDAL raster suitable for segmentation.

    Args:
        provider_name: The layer's data provider name (e.g., "gdal", "wms").
        source_path: The layer's source path (file path or URI).
        band_count: Number of bands in the layer.

    Returns:
        True if the layer is a valid 3-band GDAL-backed GeoTIFF file.
    """
    if provider_name is None or source_path is None:
        return False

    # Must be GDAL provider
    if provider_name.lower() != "gdal":
        return False

    # Must have exactly 3 bands
    if band_count != 3:
        return False

    # Extract file path without layer options (e.g., "|layername=...")
    file_path = source_path.split("|")[0]

    # Must be a .tif or .tiff extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in SUPPORTED_RASTER_EXTENSIONS:
        return False

    return True


def is_renderable_non_file_layer(
    layer_type: str,
    provider_name: Optional[str],
    source_path: Optional[str],
) -> bool:
    """Check if a layer is a renderable map layer that is not a file-backed raster.

    This includes WMS, WMTS, XYZ tiles, ArcGIS services, vector layers, etc.
    These layers can be rendered to a raster via Convert map to raster.

    Args:
        layer_type: The layer type string (e.g., "RasterLayer", "VectorLayer").
        provider_name: The layer's data provider name.
        source_path: The layer's source path or URI.

    Returns:
        True if the layer is renderable but not a file-backed GDAL raster.
    """
    if provider_name is None:
        return False

    provider_lower = provider_name.lower()

    # Explicit web service providers
    if provider_lower in WEB_SERVICE_PROVIDERS:
        return True

    # Vector layers are renderable but not rasters
    if layer_type == "VectorLayer":
        return True

    # GDAL layers with non-local sources (e.g., /vsicurl/, http://)
    if provider_lower == "gdal" and source_path:
        source_lower = source_path.lower()
        if source_lower.startswith(("/vsicurl/", "http://", "https://")):
            return True

    return False


def build_convert_map_to_raster_params(
    extent_tuple: Tuple[float, float, float, float, Optional[str]],
    layer_id_or_name: str,
    map_units_per_pixel: float = MAP_UNITS_PER_PIXEL_DEFAULT,
) -> Dict[str, Any]:
    """Build parameter dict for the Convert map to raster (native:rasterize) algorithm.

    Args:
        extent_tuple: (xmin, ymin, xmax, ymax, crs_authid) extent tuple.
            CRS authid may be None if unavailable.
        layer_id_or_name: Layer ID or name to render.
        map_units_per_pixel: Map units per pixel (default 1.0).

    Returns:
        Dictionary of parameter keys to values for the algorithm.
    """
    xmin, ymin, xmax, ymax, crs_authid = extent_tuple

    # Format extent as "xmin,xmax,ymin,ymax [crs]" for QGIS processing
    if crs_authid:
        extent_str = f"{xmin},{xmax},{ymin},{ymax} [{crs_authid}]"
    else:
        extent_str = f"{xmin},{xmax},{ymin},{ymax}"

    params = {
        "EXTENT": extent_str,
        "LAYERS": [layer_id_or_name],
        "MAP_UNITS_PER_PIXEL": map_units_per_pixel,
        # Let user choose output location; empty string opens save dialog
        "OUTPUT": "",
    }
    return params


def extract_layer_metadata(layer: Any) -> Dict[str, Any]:
    """Extract metadata from a QGIS layer object for detection logic.

    This is a thin wrapper that extracts the fields needed by the detection
    functions. Designed to be easily mockable in tests.

    Args:
        layer: A QGIS layer object (QgsMapLayer or subclass).

    Returns:
        Dictionary with keys: layer_type, provider_name, source_path, band_count.
    """
    layer_type = type(layer).__name__

    provider = getattr(layer, "dataProvider", lambda: None)()
    provider_name = provider.name() if provider else None

    source_path = getattr(layer, "source", lambda: None)()

    # band_count only makes sense for raster layers
    band_count = 0
    if hasattr(layer, "bandCount"):
        try:
            band_count = layer.bandCount()
        except Exception:
            band_count = 0

    return {
        "layer_type": layer_type,
        "provider_name": provider_name,
        "source_path": source_path,
        "band_count": band_count,
    }


def get_canvas_extent_tuple(canvas: Any) -> Tuple[float, float, float, float, Optional[str]]:
    """Extract extent tuple from a QgsMapCanvas.

    Args:
        canvas: QgsMapCanvas instance.

    Returns:
        Tuple of (xmin, ymin, xmax, ymax, crs_authid).
    """
    extent = canvas.extent()
    crs = canvas.mapSettings().destinationCrs() if hasattr(canvas, "mapSettings") else None
    crs_authid = crs.authid() if crs and hasattr(crs, "authid") else None

    return (
        extent.xMinimum(),
        extent.yMinimum(),
        extent.xMaximum(),
        extent.yMaximum(),
        crs_authid,
    )


def open_convert_map_to_raster_dialog(
    params: Dict[str, Any],
    parent: Optional[Any] = None,
) -> bool:
    """Open the Convert map to raster processing dialog with prefilled parameters.

    This function uses the QGIS Processing framework to open the algorithm dialog
    without executing it, allowing the user to adjust settings before running.

    Args:
        params: Parameter dictionary from build_convert_map_to_raster_params().
        parent: Optional parent widget for the dialog.

    Returns:
        True if the dialog was opened successfully, False otherwise.
    """
    try:
        from qgis.core import QgsApplication
        from processing.gui.AlgorithmDialog import AlgorithmDialog
    except ImportError:
        return False

    registry = QgsApplication.processingRegistry()
    algorithm = registry.algorithmById(CONVERT_MAP_TO_RASTER_ALG_ID)

    if algorithm is None:
        # Fallback: try alternative algorithm IDs
        for alt_id in ("qgis:rasterize", "gdal:rasterize_over_fixed_value"):
            algorithm = registry.algorithmById(alt_id)
            if algorithm is not None:
                break

    if algorithm is None:
        return False

    try:
        dialog = AlgorithmDialog(algorithm.create(), parent=parent)
        # Set parameters in the dialog
        for key, value in params.items():
            try:
                dialog.setParameter(key, value)
            except Exception:
                # Parameter may not be supported; continue with others
                pass
        dialog.show()
        return True
    except Exception:
        return False
