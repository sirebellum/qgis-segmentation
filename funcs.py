"""
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
"""
import math
import os
from typing import Callable, Optional, Tuple

import numpy as np


class SegmentationCanceled(Exception):
    """Raised when a segmentation task is canceled mid-flight."""


def _maybe_raise_cancel(cancel_token) -> None:
    if cancel_token is None:
        return
    checker = getattr(cancel_token, "raise_if_cancelled", None)
    if callable(checker):
        checker()
        return
    probe = getattr(cancel_token, "is_cancelled", None)
    if callable(probe) and probe():
        raise SegmentationCanceled()


def _emit_status(callback, message):
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        pass


def _require_tiff_path(source: str) -> str:
    """Ensure the path points to a GeoTIFF the runtime supports."""

    root = source.split("|")[0]
    _, ext = os.path.splitext(root)
    normalized = ext.lower()
    if normalized not in {".tif", ".tiff"}:
        raise ValueError(
            f"Raster source must be a .tif or .tiff file; got '{ext or 'unknown'}' for {root}"
        )
    return root


def _ensure_three_band(array: np.ndarray, source_label: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(
            f"{source_label} must be a 3-band channel-first array of shape (3,H,W); got shape {arr.shape}"
        )
    return np.ascontiguousarray(arr)


def _materialize_raster(raster_input):
    if isinstance(raster_input, np.ndarray):
        return _ensure_three_band(raster_input, "NumPy raster input")
    if callable(raster_input):
        materialized = raster_input()
        if not isinstance(materialized, np.ndarray):
            raise TypeError("Raster loader must return a numpy.ndarray")
        return _ensure_three_band(materialized, "Raster loader output")
    if isinstance(raster_input, str):
        source = _require_tiff_path(raster_input)
        try:
            from osgeo import gdal  # type: ignore
        except ImportError as exc:  # pragma: no cover - requires QGIS runtime
            raise RuntimeError("GDAL is required to read raster sources.") from exc
        dataset = gdal.Open(source)
        if dataset is None:
            raise RuntimeError(f"Unable to open raster source: {source}")
        bands = int(dataset.RasterCount) if dataset else 0
        if bands != 3:
            raise ValueError(f"Raster '{source}' must have exactly 3 bands; found {bands}.")
        array = dataset.ReadAsArray()
        dataset = None
        if array is None:
            raise RuntimeError("Raster source returned no data.")
        return _ensure_three_band(array, f"Raster '{source}'")
    raise TypeError(f"Unsupported raster input type: {type(raster_input)!r}")


def predict_nextgen_numpy(
    model_loader: Callable[[], object],
    array,
    num_segments: int,
    tile_size: int,
    status_callback=None,
    cancel_token=None,
):
    """Numpy-only runtime path for the monolithic next-gen model."""

    _emit_status(status_callback, "Running next-gen numpy segmentation...")
    raster = _materialize_raster(array)
    _maybe_raise_cancel(cancel_token)
    model = model_loader()
    tiles, (height_pad, width_pad), grid_shape = tile_raster(raster, tile_size)
    rows, cols = grid_shape
    h_src, w_src = raster.shape[1], raster.shape[2]
    full_h = rows * tile_size
    full_w = cols * tile_size
    canvas = np.zeros((full_h, full_w), dtype=np.uint8)
    total_tiles = tiles.shape[0]
    last_report = -1

    for idx, tile in enumerate(tiles):
        _maybe_raise_cancel(cancel_token)
        tile_float = tile.astype(np.float32, copy=False)
        labels_tile = model.predict_labels(tile_float, k=num_segments)
        r = idx // cols
        c = idx % cols
        y0 = r * tile_size
        x0 = c * tile_size
        canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = labels_tile
        percent = int(((idx + 1) / max(total_tiles, 1)) * 100)
        if percent // 5 > last_report:
            last_report = percent // 5
            _emit_status(status_callback, f"Next-gen numpy inference {percent}% complete.")

    result = canvas[: h_src, : w_src]
    _emit_status(status_callback, "Next-gen numpy segmentation complete.")
    return result


def tile_raster(array, tile_size):
    padding = lambda shape: 0 if shape % tile_size == 0 else tile_size - shape % tile_size
    channel_pad = (0, 0)
    height_pad = (0, padding(array.shape[1]))
    width_pad = (0, padding(array.shape[2]))
    array_padded = np.pad(
        array,
        (channel_pad, height_pad, width_pad),
        mode="constant",
    )

    tiles = array_padded.reshape(
        array_padded.shape[0],
        array_padded.shape[1] // tile_size,
        tile_size,
        array_padded.shape[2] // tile_size,
        tile_size,
    )
    tiles = tiles.transpose(1, 3, 0, 2, 4)
    tiles = tiles.reshape(
        tiles.shape[0] * tiles.shape[1],
        array_padded.shape[0],
        tile_size,
        tile_size,
    )

    grid_shape = (
        array_padded.shape[1] // tile_size,
        array_padded.shape[2] // tile_size,
    )

    return tiles, (height_pad[1], width_pad[1]), grid_shape


ROTATION_ALIGNMENT_THRESHOLD = 0.6
ROTATION_IMPROVEMENT_MARGIN = 0.05


def _auto_orient_tile_grid(label_map: np.ndarray, tile_size: int):
    if tile_size <= 1:
        return label_map, None
    height, width = label_map.shape
    rows = max(1, math.ceil(height / tile_size))
    cols = max(1, math.ceil(width / tile_size))
    oriented = np.zeros_like(label_map)
    rotation_plan = np.zeros((rows, cols), dtype=np.uint8)
    changed = False

    for r in range(rows):
        y0 = r * tile_size
        if y0 >= height:
            break
        y1 = min(y0 + tile_size, height)
        for c in range(cols):
            x0 = c * tile_size
            if x0 >= width:
                break
            x1 = min(x0 + tile_size, width)
            tile = label_map[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            if tile.shape[0] != tile.shape[1]:
                oriented[y0:y1, x0:x1] = tile
                rotation_plan[r, c] = 0
                continue
            top_edge = oriented[y0 - 1, x0:x1] if y0 > 0 else None
            left_edge = oriented[y0:y1, x0 - 1] if x0 > 0 else None
            best_k, best_tile = _select_tile_rotation(tile, top_edge, left_edge)
            rotation_plan[r, c] = best_k
            if best_k != 0:
                changed = True
            oriented[y0:y1, x0:x1] = best_tile

    if not changed:
        return label_map, None
    return oriented, rotation_plan


def _select_tile_rotation(tile: np.ndarray, top_edge: Optional[np.ndarray], left_edge: Optional[np.ndarray]):
    best_k = 0
    best_score = float("-inf")
    best_tile = tile
    base_score = None
    for k in range(4):
        candidate = np.rot90(tile, k=k)
        score = 0.0
        if top_edge is not None and top_edge.size > 0:
            overlap = min(candidate.shape[1], top_edge.shape[-1])
            if overlap > 0:
                score += float(np.mean(candidate[0, :overlap] == top_edge[:overlap]))
        if left_edge is not None and left_edge.size > 0:
            overlap = min(candidate.shape[0], left_edge.shape[0])
            if overlap > 0:
                score += float(np.mean(candidate[:overlap, 0] == left_edge[:overlap]))
        if k == 0:
            base_score = score
        if score > best_score + 1e-6:
            best_score = score
            best_k = k
            best_tile = candidate
    if best_k != 0:
        baseline = 0.0 if base_score is None else base_score
        if best_score < ROTATION_ALIGNMENT_THRESHOLD:
            return 0, tile
        if (best_score - baseline) < ROTATION_IMPROVEMENT_MARGIN:
            return 0, tile
    return best_k, best_tile


def _apply_rotation_plan_to_volume(volume: np.ndarray, tile_size: int, plan: np.ndarray):
    rotated = np.zeros_like(volume)
    channels, height, width = volume.shape
    rows, cols = plan.shape
    for r in range(rows):
        y0 = r * tile_size
        if y0 >= height:
            break
        y1 = min(y0 + tile_size, height)
        for c in range(cols):
            x0 = c * tile_size
            if x0 >= width:
                break
            x1 = min(x0 + tile_size, width)
            tile = volume[:, y0:y1, x0:x1]
            if tile.size == 0:
                continue
            if tile.shape[1] != tile.shape[2]:
                rotated[:, y0:y1, x0:x1] = tile
                continue
            k = int(plan[r, c]) if plan is not None else 0
            if k == 0:
                rotated[:, y0:y1, x0:x1] = tile
            else:
                rotated[:, y0:y1, x0:x1] = np.rot90(tile, k=k, axes=(-2, -1))
    return rotated


def _normalize_cluster_labels(labels, centers):
    # Validate that all label indices are within bounds
    n_centers = centers.shape[0]
    flat = labels.reshape(-1)
    if not np.all((flat >= 0) & (flat < n_centers)):
        raise ValueError(
            f"All label indices must be in [0, {n_centers-1}]. Found out-of-bounds values: {flat[(flat < 0) | (flat >= n_centers)]}"
        )
    ordering = np.argsort(centers.mean(axis=1))
    mapping = np.zeros_like(ordering)
    mapping[ordering] = np.arange(ordering.size)
    flat = mapping[flat]
    return flat.reshape(labels.shape)
