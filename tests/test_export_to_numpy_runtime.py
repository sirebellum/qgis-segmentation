# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

from pathlib import Path

import numpy as np
import pytest
import torch

from model.runtime_numpy import load_runtime_model
from training.config import Config
from training.export import export_numpy_artifacts


def _dummy_state_dict():
    return {
        "encoder.stem.0.weight": torch.zeros((2, 3, 3, 3)),
        "encoder.stem.0.bias": torch.zeros((2,)),
        "encoder.stem.1.weight": torch.ones((2,)),
        "encoder.stem.1.bias": torch.zeros((2,)),
        "encoder.stem.1.running_mean": torch.zeros((2,)),
        "encoder.stem.1.running_var": torch.ones((2,)),
        "encoder.stem.3.weight": torch.zeros((4, 2, 3, 3)),
        "encoder.stem.3.bias": torch.zeros((4,)),
        "encoder.stem.4.weight": torch.ones((4,)),
        "encoder.stem.4.bias": torch.zeros((4,)),
        "encoder.stem.4.running_mean": torch.zeros((4,)),
        "encoder.stem.4.running_var": torch.ones((4,)),
        "encoder.block1.conv1.weight": torch.zeros((4, 4, 3, 3)),
        "encoder.block1.conv1.bias": torch.zeros((4,)),
        "encoder.block1.bn1.weight": torch.ones((4,)),
        "encoder.block1.bn1.bias": torch.zeros((4,)),
        "encoder.block1.bn1.running_mean": torch.zeros((4,)),
        "encoder.block1.bn1.running_var": torch.ones((4,)),
        "encoder.block1.conv2.weight": torch.zeros((4, 4, 3, 3)),
        "encoder.block1.conv2.bias": torch.zeros((4,)),
        "encoder.block1.bn2.weight": torch.ones((4,)),
        "encoder.block1.bn2.bias": torch.zeros((4,)),
        "encoder.block1.bn2.running_mean": torch.zeros((4,)),
        "encoder.block1.bn2.running_var": torch.ones((4,)),
        "encoder.block2.conv1.weight": torch.zeros((4, 4, 3, 3)),
        "encoder.block2.conv1.bias": torch.zeros((4,)),
        "encoder.block2.bn1.weight": torch.ones((4,)),
        "encoder.block2.bn1.bias": torch.zeros((4,)),
        "encoder.block2.bn1.running_mean": torch.zeros((4,)),
        "encoder.block2.bn1.running_var": torch.ones((4,)),
        "encoder.block2.conv2.weight": torch.zeros((4, 4, 3, 3)),
        "encoder.block2.conv2.bias": torch.zeros((4,)),
        "encoder.block2.bn2.weight": torch.ones((4,)),
        "encoder.block2.bn2.bias": torch.zeros((4,)),
        "encoder.block2.bn2.running_mean": torch.zeros((4,)),
        "encoder.block2.bn2.running_var": torch.ones((4,)),
        "cluster_head.seed_proj.weight": torch.zeros((4, 4, 1, 1)),
        "cluster_head.seed_proj.bias": torch.zeros((4,)),
    }


def test_export_and_load_runtime(tmp_path):
    cfg = Config()
    state = _dummy_state_dict()
    out_dir = tmp_path / "best"
    mirror = tmp_path / "mirror"

    export_numpy_artifacts(state, cfg, score=1.0, step=1, out_dir=str(out_dir), mirror_dir=str(mirror))

    runtime = load_runtime_model(str(out_dir))
    rgb = np.ones((3, 32, 32), dtype=np.float32)
    labels = runtime.predict_labels(rgb, k=4)

    assert labels.shape == (32, 32)
    assert labels.dtype == np.uint8
    assert labels.max() < 4
    assert (mirror / "model.npz").exists()
    assert (mirror / "meta.json").exists()


def test_runtime_probabilities_sum_to_one(tmp_path):
    cfg = Config()
    out_dir = tmp_path / "best"
    export_numpy_artifacts(_dummy_state_dict(), cfg, score=0.0, step=0, out_dir=str(out_dir))
    runtime = load_runtime_model(str(out_dir))
    rgb = np.zeros((3, 8, 8), dtype=np.float32)
    probs = runtime.forward(rgb, k=2)

    per_pixel = probs.sum(axis=0)
    assert per_pixel.shape == (8, 8)
    assert np.allclose(per_pixel, 1.0, atol=1e-5)