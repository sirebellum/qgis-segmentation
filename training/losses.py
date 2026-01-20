# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Unsupervised losses for two-view training."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .losses_distill import boundary_prior_losses

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


def edge_aware_smoothness(probs: torch.Tensor, rgb: torch.Tensor, weight: float) -> torch.Tensor:
    grad_p = image_gradients(probs)
    grad_rgb = image_gradients(rgb)
    edge = grad_rgb.mean(dim=1, keepdim=True)
    attn = torch.exp(-weight * edge)
    return (attn * grad_p.pow(2)).mean()


def total_loss(
    cfg: LossConfig,
    probs1: torch.Tensor,
    probs2: torch.Tensor,
    rgb1: torch.Tensor,
    rgb2: torch.Tensor,
    grid: torch.Tensor,
    slic1: torch.Tensor | None = None,
    slic2: torch.Tensor | None = None,
    emb1: torch.Tensor | None = None,
    emb2: torch.Tensor | None = None,
) -> dict:
    cons = two_view_consistency(probs1, probs2, grid)
    ent_pixel1, ent_marg1 = entropy_terms(probs1)
    ent_pixel2, ent_marg2 = entropy_terms(probs2)
    ent_pixel = 0.5 * (ent_pixel1 + ent_pixel2)
    ent_marg = 0.5 * (ent_marg1 + ent_marg2)
    smooth = 0.5 * (
        edge_aware_smoothness(probs1, rgb1, cfg.edge_weight)
        + edge_aware_smoothness(probs2, rgb2, cfg.edge_weight)
    )

    loss = (
        cfg.consistency_weight * cons
        + cfg.entropy_min_weight * ent_pixel
        - cfg.entropy_marginal_weight * ent_marg
        + cfg.smoothness_weight * smooth
    )
    boundary = torch.tensor(0.0, device=loss.device)
    antimerge = torch.tensor(0.0, device=loss.device)
    smooth_within = torch.tensor(0.0, device=loss.device)
    if (
        slic1 is not None
        and slic2 is not None
        and emb1 is not None
        and emb2 is not None
        and (cfg.slic_boundary_weight > 0 or cfg.slic_antimerge_weight > 0 or cfg.slic_within_weight > 0)
    ):
        priors1 = boundary_prior_losses(
            emb1,
            slic1,
            rgb1,
            lambda_boundary=cfg.slic_boundary_weight,
            lambda_antimerge=cfg.slic_antimerge_weight,
            lambda_within=cfg.slic_within_weight,
        )
        priors2 = boundary_prior_losses(
            emb2,
            slic2,
            rgb2,
            lambda_boundary=cfg.slic_boundary_weight,
            lambda_antimerge=cfg.slic_antimerge_weight,
            lambda_within=cfg.slic_within_weight,
        )
        boundary = 0.5 * (priors1["boundary"] + priors2["boundary"])
        antimerge = 0.5 * (priors1["antimerge"] + priors2["antimerge"])
        smooth_within = 0.5 * (priors1["smooth_within"] + priors2["smooth_within"])
        loss = loss + boundary + antimerge + smooth_within

    return {
        "loss": loss,
        "consistency": cons.detach(),
        "entropy_pixel": ent_pixel.detach(),
        "entropy_marginal": ent_marg.detach(),
        "smoothness": smooth.detach(),
        "boundary": boundary.detach(),
        "antimerge": antimerge.detach(),
        "smooth_within": smooth_within.detach(),
    }
