# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from model.runtime_numpy import RUNTIME_META_VERSION, load_runtime_model
from training.export import smoke_export_runtime


def _gradient_rgb(height: int, width: int) -> np.ndarray:
    base = np.arange(height * width, dtype=np.float32).reshape(height, width)
    rgb = np.stack([base, base + 1.0, base + 2.0], axis=0)
    return rgb


def test_smoke_export_round_trip(tmp_path):
    out_dir = tmp_path / "artifact"
    smoke_export_runtime(out_dir=str(out_dir), seed=11, steps=1, embed_dim=8, max_k=4, patch_size=32)

    runtime = load_runtime_model(str(out_dir))
    rgb = _gradient_rgb(16, 18)

    labels = runtime.predict_labels(rgb, k=3)
    assert labels.shape == (16, 18)
    assert labels.dtype == np.uint8
    assert labels.max() < runtime.max_k

    probs = runtime.forward(rgb, k=3)
    assert probs.shape[1:] == labels.shape
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-4)


def test_runtime_meta_schema_enforced(tmp_path):
    out_dir = tmp_path / "artifact"
    smoke_export_runtime(out_dir=str(out_dir), seed=5, steps=1, embed_dim=6, max_k=3, patch_size=24)

    meta_path = Path(out_dir) / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["version"] == RUNTIME_META_VERSION
    assert len(meta["input_mean"]) == 3
    assert len(meta["input_std"]) == 3
    assert meta["input_scale"] > 0
    assert meta["max_k"] >= 2

    bad_dir = tmp_path / "bad_meta"
    shutil.copytree(out_dir, bad_dir)
    bad_meta = json.loads((bad_dir / "meta.json").read_text())
    bad_meta.pop("input_scale", None)
    (bad_dir / "meta.json").write_text(json.dumps(bad_meta), encoding="utf-8")

    with pytest.raises(ValueError):
        load_runtime_model(str(bad_dir))
