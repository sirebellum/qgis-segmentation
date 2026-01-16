# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Shape and contract tests for the monolithic model."""
from __future__ import annotations

import pytest
import torch

from training.config import default_config
from training.models.model import MonolithicSegmenter


@pytest.mark.parametrize("k", [2, 4, 8, 16])
@pytest.mark.parametrize("with_elev", [False, True])
@pytest.mark.parametrize("smoothing_lane", ["fast", "learned", "none"])
def test_forward_shapes(k, with_elev, smoothing_lane):
    cfg = default_config()
    model = MonolithicSegmenter(cfg.model)
    rgb = torch.rand(1, 3, 512, 512)
    elev = torch.rand(1, 1, 512, 512) if with_elev else None
    out = model(
        rgb,
        k=k,
        elev=elev,
        elev_present=with_elev,
        downsample=1,
        cluster_iters=2,
        smooth_iters=1,
        smoothing_lane=smoothing_lane,
    )
    probs = out["probs"]
    assert probs.shape == (1, k, 512, 512)
    prob_sum = probs.sum(dim=1)
    assert torch.isfinite(prob_sum).all()
    assert torch.all(prob_sum >= 0)
    # Normalization may be relaxed after smoothing; ensure sums stay bounded.
    assert torch.all(prob_sum <= 1.05)
