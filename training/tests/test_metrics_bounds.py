# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import torch
import pytest

from training.metrics import boundary_density, cluster_utilization, speckle_score, view_consistency_score


def test_cluster_utilization_matches_uniform_entropy():
    probs = torch.full((1, 4, 2, 2), 0.25)
    util = cluster_utilization(probs)
    expected = torch.log(torch.tensor(4.0))
    assert pytest.approx(expected.item(), rel=0.05) == util.item()


def test_speckle_and_boundary_scores_zero_for_constant_map():
    labels = torch.zeros((1, 2, 3), dtype=torch.int64)
    assert speckle_score(labels) == 0
    assert boundary_density(labels).item() < 1e-3


def test_view_consistency_score_bounds():
    p1 = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    p2 = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    same = view_consistency_score(p1, p2)
    assert same.item() == pytest.approx(1.0, rel=1e-3)

    p3 = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    opposite = view_consistency_score(p1, p3)
    assert opposite.item() == pytest.approx(0.0, abs=1e-3)
