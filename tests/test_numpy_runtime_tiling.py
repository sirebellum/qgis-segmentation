# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import numpy as np

from funcs import predict_nextgen_numpy


class _StubRuntime:
    def __init__(self, fill_value: int = 0):
        self.fill_value = int(fill_value)

    def predict_labels(self, tile: np.ndarray, k: int):
        height, width = tile.shape[1], tile.shape[2]
        value = min(max(k - 1, 0), 255)
        return np.full((height, width), value, dtype=np.uint8)


def test_predict_nextgen_numpy_stitches_tiles():
    raster = np.ones((3, 33, 65), dtype=np.float32)
    runtime = _StubRuntime()

    labels = predict_nextgen_numpy(lambda: runtime, raster, num_segments=3, tile_size=16)

    assert labels.shape == raster.shape[1:]
    assert labels.dtype == np.uint8
    assert labels.max() == 2
    assert labels.min() == 2