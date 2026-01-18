# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import json
from pathlib import Path

import numpy as np
import pytest

from model.runtime_numpy import REQUIRED_META_KEYS, load_runtime_artifacts, load_runtime_model
from training.export import smoke_export_runtime


def test_smoke_export_runtime_contract(tmp_path):
    out_dir = tmp_path / "artifact"
    smoke_export_runtime(out_dir=str(out_dir), seed=9, steps=1, embed_dim=6, max_k=4, patch_size=24)

    weights, meta = load_runtime_artifacts(str(out_dir))
    assert REQUIRED_META_KEYS.issubset(set(meta.__dict__.keys()))
    assert meta.version and meta.max_k >= 2
    assert meta.embed_dim == 6

    runtime = load_runtime_model(str(out_dir))
    rgb = np.linspace(0, 1, num=48, dtype=np.float32).reshape(3, 4, 4)
    probs = runtime.forward(rgb, k=3)
    assert probs.shape == (3, 4, 4)
    assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-3)

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert "score" in metrics and "step" in metrics


def test_missing_weight_raises(tmp_path):
    src = tmp_path / "good"
    smoke_export_runtime(out_dir=str(src), seed=5, steps=1, embed_dim=4, max_k=3, patch_size=16)

    broken = tmp_path / "broken"
    broken.mkdir()
    meta = json.loads((src / "meta.json").read_text())
    (broken / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    with np.load(src / "model.npz") as handle:
        payload = {k: handle[k] for k in handle.files if k != "seed_proj.weight"}
    np.savez(broken / "model.npz", **payload)

    with pytest.raises(ValueError):
        load_runtime_model(str(broken))
