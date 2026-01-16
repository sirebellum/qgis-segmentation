# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""CPU smoke train for 1–2 steps on synthetic data."""
from __future__ import annotations

import torch

from training.config import default_config
from training.data.dataset import UnsupervisedRasterDataset
from training.data.synthetic import SyntheticDataset
from training.losses import total_loss
from training.models.model import MonolithicSegmenter


def test_smoke_train_step():
    cfg = default_config()
    cfg.train.steps = 2
    cfg.train.batch_size = 1
    cfg.train.amp = False
    base = SyntheticDataset(num_samples=2, cfg=cfg.data)
    ds = UnsupervisedRasterDataset(base.samples, cfg.data, cfg.aug)
    model = MonolithicSegmenter(cfg.model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    for step in range(cfg.train.steps):
        item = ds[step % len(ds)]
        v1 = item["view1"]
        v2 = item["view2"]
        grid = item["warp_grid"]
        rgb1 = v1["rgb"]
        rgb2 = v2["rgb"]
        out1 = model(rgb1, k=4, cluster_iters=2)
        out2 = model(rgb2, k=4, cluster_iters=2)
        losses = total_loss(cfg.loss, out1["probs"], out2["probs"], rgb1, rgb2, grid)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        opt.step()

    assert True
