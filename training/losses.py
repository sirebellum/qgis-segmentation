# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Unsupervised losses for two-view training."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import LossConfig
from .utils.gradients import image_gradients
from .utils.warp import apply_warp


def two_view_consistency(p1: torch.Tensor, p2: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between two probability maps after warping p2 to p1 frame."""
    p2_warp = apply_warp(p2, grid)
    kl1 = F.kl_div(p2_warp.log().clamp(min=-10), p1, reduction="batchmean")
    kl2 = F.kl_div(p1.log().clamp(min=-10), p2_warp, reduction="batchmean")
    return 0.5 * (kl1 + kl2)


def entropy_terms(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = probs.clamp(min=1e-8)
    pixel_entropy = -(flat * flat.log()).sum(dim=1).mean()
    marginal = flat.mean(dim=(0, 2, 3))
    marginal_entropy = -(marginal * marginal.clamp(min=1e-8).log()).sum()
    return pixel_entropy, marginal_entropy


def edge_aware_smoothness(probs: torch.Tensor, rgb: torch.Tensor, elev: torch.Tensor | None, weight: float) -> torch.Tensor:
    grad_p = image_gradients(probs)
    grad_rgb = image_gradients(rgb)
    edge = grad_rgb.mean(dim=1, keepdim=True)
    if elev is not None:
        edge = edge + 0.5 * image_gradients(elev)
    attn = torch.exp(-weight * edge)
    return (attn * grad_p.pow(2)).mean()


def total_loss(
    cfg: LossConfig,
    probs1: torch.Tensor,
    probs2: torch.Tensor,
    rgb1: torch.Tensor,
    rgb2: torch.Tensor,
    grid: torch.Tensor,
    elev1: torch.Tensor | None = None,
    elev2: torch.Tensor | None = None,
) -> dict:
    cons = two_view_consistency(probs1, probs2, grid)
    ent_pixel1, ent_marg1 = entropy_terms(probs1)
    ent_pixel2, ent_marg2 = entropy_terms(probs2)
    ent_pixel = 0.5 * (ent_pixel1 + ent_pixel2)
    ent_marg = 0.5 * (ent_marg1 + ent_marg2)
    smooth = 0.5 * (
        edge_aware_smoothness(probs1, rgb1, elev1, cfg.edge_weight)
        + edge_aware_smoothness(probs2, rgb2, elev2, cfg.edge_weight)
    )

    loss = (
        cfg.consistency_weight * cons
        + cfg.entropy_min_weight * ent_pixel
        - cfg.entropy_marginal_weight * ent_marg
        + cfg.smoothness_weight * smooth
    )
    return {
        "loss": loss,
        "consistency": cons.detach(),
        "entropy_pixel": ent_pixel.detach(),
        "entropy_marginal": ent_marg.detach(),
        "smoothness": smooth.detach(),
    }
