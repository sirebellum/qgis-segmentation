# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Synthetic dataset checks."""
from __future__ import annotations

from training.config import default_config
from training.data.dataset import UnsupervisedRasterDataset
from training.data.synthetic import SyntheticDataset


def test_synthetic_dataset_shapes():
    cfg = default_config()
    base = SyntheticDataset(num_samples=2, with_elevation=True, seed=7)
    ds = UnsupervisedRasterDataset(base.samples, cfg.data, cfg.aug)
    item = ds[0]
    assert "view1" in item and "view2" in item
    assert item["view1"]["rgb"].shape[-2:] == (512, 512)
