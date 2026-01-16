# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

from pathlib import Path

import numpy as np
import pytest

from segmenter.model import runtime_backend
from segmenter.model.runtime_numpy import load_runtime_model
from training.export import smoke_export_runtime


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def test_auto_falls_back_to_numpy_when_torch_missing(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(runtime_backend, "_import_torch", lambda: None)
    runtime = runtime_backend.load_runtime(root / "model" / "best", prefer="auto")
    assert getattr(runtime, "backend") == "numpy"
    assert runtime.device_label == "cpu"


def test_prefer_numpy_uses_numpy_even_if_torch_present(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(runtime_backend, "_import_torch", lambda: object())
    runtime = runtime_backend.load_runtime(root / "model" / "best", prefer="numpy")
    assert getattr(runtime, "backend") == "numpy"


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_torch_cpu_backend_runs(tmp_path):
    out_dir = tmp_path / "artifact"
    smoke_export_runtime(out_dir=str(out_dir), seed=13, steps=1, embed_dim=4, max_k=3, patch_size=16)
    runtime = runtime_backend.load_runtime(out_dir, prefer="torch", device_preference="cpu")

    rgb = np.ones((3, 8, 8), dtype=np.float32)
    labels = runtime.predict_labels(rgb, k=3)

    assert getattr(runtime, "backend") == "torch"
    assert "cpu" in str(getattr(runtime, "device_label", "")).lower()
    assert labels.shape == (8, 8)
    assert labels.dtype == np.uint8


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_torch_backend_matches_numpy_shape(tmp_path):
    out_dir = tmp_path / "artifact"
    smoke_export_runtime(out_dir=str(out_dir), seed=5, steps=1, embed_dim=4, max_k=3, patch_size=12)

    torch_runtime = runtime_backend.load_runtime(out_dir, prefer="torch", device_preference="cpu")
    numpy_runtime = load_runtime_model(str(out_dir))

    rgb = np.random.rand(3, 10, 11).astype(np.float32)
    torch_labels = torch_runtime.predict_labels(rgb, k=3)
    numpy_labels = numpy_runtime.predict_labels(rgb, k=3)

    assert torch_labels.shape == numpy_labels.shape
    assert torch_labels.dtype == numpy_labels.dtype == np.uint8
