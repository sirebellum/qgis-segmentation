from __future__ import annotations

import numpy as np


def ensure_channel_first(array: np.ndarray) -> np.ndarray:
    """Ensure raster data is channel-first (C, H, W).

    Accepts either 2D arrays (H, W) or 3D arrays already in channel-first
    order. Returns a contiguous copy to make downstream GDAL writes safe.
    """

    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D input")
    return np.ascontiguousarray(arr)
