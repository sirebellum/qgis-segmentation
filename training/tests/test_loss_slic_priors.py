# SPDX-License-Identifier: BSD-3-Clause
import torch

from training.config import LossConfig
from training.losses import total_loss


def test_total_loss_handles_slic_priors():
    cfg = LossConfig(
        consistency_weight=0.0,
        entropy_min_weight=0.0,
        entropy_marginal_weight=0.0,
        smoothness_weight=0.0,
        edge_weight=1.0,
        slic_boundary_weight=1.0,
        slic_antimerge_weight=0.5,
        slic_within_weight=0.1,
    )
    # two-class probabilities with clear boundaries
    probs1 = torch.tensor(
        [[[[0.8, 0.8], [0.2, 0.2]], [[0.2, 0.2], [0.8, 0.8]]]], dtype=torch.float32
    )
    probs2 = probs1.clone()
    rgb1 = torch.ones(1, 3, 2, 2) * 0.5
    rgb2 = rgb1.clone()
    grid = torch.zeros(1, 2, 2, 2)

    # embeddings at stride-1 for simplicity
    emb1 = torch.ones(1, 4, 2, 2)
    emb2 = torch.ones(1, 4, 2, 2)
    slic1 = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.int64)
    slic2 = slic1.clone()

    losses = total_loss(
        cfg,
        probs1,
        probs2,
        rgb1,
        rgb2,
        grid,
        slic1=slic1,
        slic2=slic2,
        emb1=emb1,
        emb2=emb2,
    )

    assert torch.isfinite(losses["loss"]).all()
    assert torch.isfinite(losses["boundary"]).all()
    assert losses["boundary"] > 0
    assert losses["antimerge"] >= 0
    assert losses["smooth_within"] >= 0
