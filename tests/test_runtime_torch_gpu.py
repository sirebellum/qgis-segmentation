# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import os
from pathlib import Path

import numpy as np
import pytest

from segmenter.model import runtime_backend
from training.export import smoke_export_runtime


def _has_torch_gpu() -> bool:
    try:
        import torch
    except Exception:
        return False
    if torch.cuda.is_available():
        return True
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS", "0") not in {"1", "true", "TRUE"}, reason="GPU tests disabled")
@pytest.mark.skipif(not _has_torch_gpu(), reason="No GPU-capable torch device available")
def test_torch_gpu_backend(tmp_path):
    out_dir = tmp_path / "artifact"
    smoke_export_runtime(out_dir=str(out_dir), seed=17, steps=1, embed_dim=4, max_k=3, patch_size=12)

    runtime = runtime_backend.load_runtime(out_dir, prefer="torch", device_preference="auto")
    rgb = np.random.rand(3, 12, 9).astype(np.float32)
    labels = runtime.predict_labels(rgb, k=3)

    assert getattr(runtime, "backend") == "torch"
    assert labels.shape == (12, 9)
    assert labels.dtype == np.uint8
    assert "cpu" not in str(getattr(runtime, "device_label", "cpu")).lower()
