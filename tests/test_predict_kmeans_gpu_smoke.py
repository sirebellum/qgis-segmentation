# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import os

import numpy as np
import pytest
import torch

import funcs


@pytest.mark.gpu
def test_predict_kmeans_gpu_smoke():
    if os.environ.get("RUN_GPU_TESTS") != "1":
        pytest.skip("GPU tests disabled via RUN_GPU_TESTS")

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        pytest.skip("No CUDA or MPS backend available for GPU smoke test.")

    rng = np.random.default_rng(7)
    array = rng.random((3, 64, 64), dtype=np.float32)

    result = funcs.predict_kmeans(array, num_segments=3, resolution=8, device_hint=device)

    assert result.shape == array.shape[1:]
    assert result.max() < 3
