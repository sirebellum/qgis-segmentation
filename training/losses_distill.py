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


def _downsample_labels(labels: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if labels.dim() != 3:
        raise ValueError("labels must be [B,H,W]")
    return F.interpolate(labels.unsqueeze(1).float(), size=size, mode="nearest").long().squeeze(1)


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


def boundary_prior_losses(
    emb: torch.Tensor,
    slic_labels: torch.Tensor,
    rgb: torch.Tensor,
    *,
    lambda_boundary: float = 1.0,
    lambda_antimerge: float = 1.0,
    lambda_within: float = 0.1,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Compute SLIC-guided boundary and anti-merge penalties.

    Args:
        emb: student embeddings [B,C,Hs,Ws] (stride-4 grid).
        slic_labels: precomputed SLIC labels [B,H,W] at input resolution.
        rgb: input RGB tensor [B,3,H,W] normalized to [0,1].
        lambda_boundary: weight for encouraging boundaries (lower similarity across SLIC edges).
        lambda_antimerge: weight for edge-strength-weighted separation across SLIC edges.
        lambda_within: optional smoothness inside superpixels (small, edge-gated).
    """

    emb_norm = l2_normalize(emb, dim=1)
    target_size = emb.shape[-2:]
    slic_ds = _downsample_labels(slic_labels, target_size)
    rgb_ds = F.interpolate(rgb, size=target_size, mode="bilinear", align_corners=False)

    sim_x = (emb_norm[:, :, :, 1:] * emb_norm[:, :, :, :-1]).sum(dim=1)
    sim_y = (emb_norm[:, :, 1:, :] * emb_norm[:, :, :-1, :]).sum(dim=1)

    boundary_x = (slic_ds[:, :, 1:] != slic_ds[:, :, :-1]).float()
    boundary_y = (slic_ds[:, 1:, :] != slic_ds[:, :-1, :]).float()
    within_x = 1.0 - boundary_x
    within_y = 1.0 - boundary_y

    grad_x = (rgb_ds[:, :, :, 1:] - rgb_ds[:, :, :, :-1]).abs().mean(dim=1)
    grad_y = (rgb_ds[:, :, 1:, :] - rgb_ds[:, :, :-1, :]).abs().mean(dim=1)

    boundary_loss = lambda_boundary * (
        (sim_x * boundary_x).mean() + (sim_y * boundary_y).mean()
    )
    antimerge_loss = lambda_antimerge * (
        (sim_x * boundary_x * grad_x).mean() + (sim_y * boundary_y * grad_y).mean()
    )

    smooth_within = 0.0
    if lambda_within > 0:
        weight_x = (1.0 / (grad_x + eps)).clamp(max=10.0)
        weight_y = (1.0 / (grad_y + eps)).clamp(max=10.0)
        smooth_within = lambda_within * (
            ((1.0 - sim_x) * within_x * weight_x).mean() + ((1.0 - sim_y) * within_y * weight_y).mean()
        )

    return {
        "boundary": boundary_loss,
        "antimerge": antimerge_loss,
        "smooth_within": smooth_within if isinstance(smooth_within, torch.Tensor) else torch.tensor(smooth_within, device=emb.device),
    }


def _update_ema(ema_state: Dict[str, torch.Tensor], key: str, value: torch.Tensor, decay: float, eps: float) -> torch.Tensor:
    if key not in ema_state:
        ema_state[key] = value.detach()
    else:
        ema_state[key] = decay * ema_state[key] + (1 - decay) * value.detach()
    return ema_state[key].clamp(min=eps)


def scale_neutral_merge(
    losses: Dict[str, torch.Tensor],
    ema_state: Dict[str, torch.Tensor],
    *,
    decay: float = 0.9,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Merge per-scale losses via EMA-normalized geometric mean.

    Each loss term is divided by its running EMA (tracked in-place in ``ema_state``),
    log-averaged, and exponentiated to avoid preferring any specific scale
    (e.g., patch sizes 256/512/1024).
    """

    if not losses:
        zero = torch.tensor(0.0)
        return zero, ema_state
    log_terms = []
    for key in sorted(losses.keys()):
        ema_val = _update_ema(ema_state, key, losses[key], decay, eps)
        normalized = torch.log((losses[key] / (ema_val + eps)).clamp(min=eps))
        log_terms.append(normalized)
    merged = torch.exp(torch.stack(log_terms).mean())
    return merged, ema_state


def normalize_with_ema(value: torch.Tensor, ema_state: Dict[str, torch.Tensor], key: str, decay: float, eps: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ema_val = _update_ema(ema_state, key, value, decay, eps)
    return value / (ema_val + eps), ema_state


def cross_resolution_consistency(assignments: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Penalize disagreement between adjacent resolution assignments."""

    if not assignments:
        return torch.tensor(0.0)
    device = next(iter(assignments.values())).device
    losses = []
    if "coarse" in assignments and "mid" in assignments:
        up = F.interpolate(assignments["coarse"], size=assignments["mid"].shape[-2:], mode="bilinear", align_corners=False)
        losses.append(F.mse_loss(up, assignments["mid"]))
    if "mid" in assignments and "fine" in assignments:
        up = F.interpolate(assignments["mid"], size=assignments["fine"].shape[-2:], mode="bilinear", align_corners=False)
        losses.append(F.mse_loss(up, assignments["fine"]))
    if not losses:
        return torch.tensor(0.0, device=device)
    return sum(losses) / len(losses)


def geometric_mean_normalized(values: Dict[str, torch.Tensor], eps: float = 1e-4) -> torch.Tensor:
    """Compute a symmetric geometric mean across normalized loss values."""

    if not values:
        return torch.tensor(0.0)
    terms = [torch.log(v.clamp(min=eps)) for v in values.values()]
    return torch.exp(torch.stack(terms).mean())
