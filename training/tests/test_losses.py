# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Loss sanity checks."""
from __future__ import annotations

import torch

from training.config import default_config
from training.losses import total_loss
from training.models.model import MonolithicSegmenter


def test_losses_finite_backward():
    cfg = default_config()
    model = MonolithicSegmenter(cfg.model)
    rgb = torch.rand(1, 3, 512, 512)
    out1 = model(rgb, k=4, elev_present=False)
    out2 = model(rgb, k=4, elev_present=False)
    grid = torch.zeros(1, 512, 512, 2)
    losses = total_loss(cfg.loss, out1["probs"], out2["probs"], rgb, rgb, grid)
    loss = losses["loss"]
    assert torch.isfinite(loss)
    loss.backward()
