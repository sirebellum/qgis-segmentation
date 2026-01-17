# SPDX-License-Identifier: BSD-3-Clause
"""Distillation + clustering losses for teacher→student training."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.nn import functional as F

from .models.student_cnn import affinity_matrix, batched_kmeans, l2_normalize


def feature_distillation(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    s = l2_normalize(student, dim=1)
    t = l2_normalize(teacher, dim=1)
    return 1.0 - (s * t).sum(dim=1).mean()


def affinity_distillation(student: torch.Tensor, teacher: torch.Tensor, sample: int = 512) -> torch.Tensor:
    sim_s = affinity_matrix(student, sample=sample)
    sim_t = affinity_matrix(teacher, sample=sample)
    return F.mse_loss(sim_s, sim_t)


def clustering_losses(emb: torch.Tensor, k: int, iters: int = 5, temperature: float = 0.8) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    assign, proto = batched_kmeans(emb, k=k, iters=iters, temperature=temperature)
    flat_assign = assign.view(assign.shape[0], assign.shape[1], -1)
    hist = flat_assign.mean(dim=2)
    balance = (hist * (hist + 1e-6).log()).sum(dim=1).mean()
    sharp = - (flat_assign * (flat_assign + 1e-6).log()).sum(dim=1).mean()
    proto_norm = proto.norm(dim=2).mean()
    pull = proto_norm  # encourage non-collapse
    loss = sharp + balance + 0.1 * pull
    return loss, {"assign": assign.detach(), "proto": proto.detach()}


def edge_aware_tv(assign: torch.Tensor, rgb: torch.Tensor, weight: float = 2.0) -> torch.Tensor:
    gx = rgb[:, :, :, 1:] - rgb[:, :, :, :-1]
    gy = rgb[:, :, 1:, :] - rgb[:, :, :-1, :]
    ax = assign[:, :, :, 1:] - assign[:, :, :, :-1]
    ay = assign[:, :, 1:, :] - assign[:, :, :-1, :]
    edge_x = torch.exp(-weight * gx.abs().mean(dim=1, keepdim=True))
    edge_y = torch.exp(-weight * gy.abs().mean(dim=1, keepdim=True))
    tv = (edge_x * ax.pow(2)).mean() + (edge_y * ay.pow(2)).mean()
    return tv
