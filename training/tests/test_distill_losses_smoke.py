# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import torch

from training.losses_distill import affinity_distillation, clustering_losses, edge_aware_tv, feature_distillation


def test_feature_and_affinity_distillation_are_finite():
    student = torch.randn(2, 8, 4, 4, requires_grad=True)
    teacher = torch.randn(2, 8, 4, 4, requires_grad=True)

    feat_loss = feature_distillation(student, teacher)
    aff_loss = affinity_distillation(student, teacher, sample=8)

    assert torch.isfinite(feat_loss)
    assert torch.isfinite(aff_loss)
    feat_loss.backward(retain_graph=True)
    aff_loss.backward()


def test_clustering_losses_and_edge_tv_shapes():
    emb = torch.randn(1, 6, 3, 3, requires_grad=True)
    loss, extras = clustering_losses(emb, k=3, iters=2, temperature=0.9)
    assert torch.isfinite(loss)
    assert "assign" in extras and "proto" in extras
    assign = extras["assign"]
    assert assign.shape == (1, 3, 3, 3)
    assert torch.all(assign >= 0)

    rgb = torch.rand(1, 3, 3, 3, requires_grad=True)
    tv = edge_aware_tv(assign, rgb, weight=1.5)
    assert torch.isfinite(tv)
    if tv.requires_grad:
        tv.backward()
