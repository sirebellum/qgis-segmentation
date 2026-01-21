# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import numpy as np
import torch

import funcs


def test_predict_kmeans_is_deterministic_with_fixed_seeds():
    rng = np.random.default_rng(42)
    array = rng.random((3, 64, 64), dtype=np.float32)

    torch.manual_seed(0)
    first = funcs.predict_kmeans(array, num_segments=5, resolution=16)

    torch.manual_seed(0)
    second = funcs.predict_kmeans(array, num_segments=5, resolution=16)

    np.testing.assert_array_equal(first, second)
