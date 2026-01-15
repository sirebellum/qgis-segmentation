# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Proxy metrics for unsupervised segmentation."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils.gradients import image_gradients


def cluster_utilization(probs: torch.Tensor) -> torch.Tensor:
    marginal = probs.mean(dim=(0, 2, 3)).clamp(min=1e-8)
    return -(marginal * marginal.log()).sum()


def speckle_score(labels: torch.Tensor) -> torch.Tensor:
    one_hot = F.one_hot(labels, num_classes=int(labels.max().item() + 1)).float()
    one_hot = one_hot.permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(one_hot, kernel_size=3, stride=1, padding=1)
    var = ((one_hot - pooled) ** 2).mean()
    return var


def boundary_density(labels: torch.Tensor) -> torch.Tensor:
    grad = image_gradients(labels.float())
    return grad.mean()


def view_consistency_score(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(p1, p2, dim=1).mean()
