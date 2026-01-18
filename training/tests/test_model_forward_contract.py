# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import pytest
import torch

from training.config import default_config
from training.models.model import MonolithicSegmenter


@pytest.mark.parametrize("k,lane", [(2, "fast"), (8, "learned"), (16, "fast")])
def test_forward_contract_shapes(k, lane):
    cfg = default_config()
    cfg.model.embed_dim = 16
    cfg.model.max_k = 16
    model = MonolithicSegmenter(cfg.model)
    rgb = torch.rand(1, 3, 64, 64)

    out = model(
        rgb,
        k=k,
        downsample=1,
        cluster_iters=2,
        smooth_iters=1,
        smoothing_lane=lane,
    )

    assert set(out.keys()) >= {"probs", "logits", "embeddings", "prototypes", "logits_latent"}
    assert out["embeddings"].shape == (1, cfg.model.embed_dim, 16, 16)
    assert out["prototypes"].shape == (1, k, cfg.model.embed_dim)
    assert out["logits_latent"].shape == (1, k, 16, 16)

    probs = out["probs"]
    assert probs.shape == (1, k, 64, 64)
    assert torch.isfinite(probs).all()
    per_pixel = probs.sum(dim=1)
    assert torch.all(per_pixel > 0)
    assert torch.all(per_pixel < 1.1)

    if lane == "learned" and k < cfg.model.max_k:
        # ensure padding did not leak extra classes
        assert probs.shape[1] == k
