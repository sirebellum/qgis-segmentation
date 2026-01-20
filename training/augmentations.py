# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic training-time augmentations for RGB + aligned maps.

Applies geometry (rotate/flip) across all spatial fields and photometric
noise/contrast/saturation to RGB only. Geometry is synchronized across RGB,
precomputed SLIC labels, and optional target maps so that alignment is
preserved. Randomness can be made deterministic by supplying a base seed to
``make_rng``; view-specific seeds keep multi-view samples independent but
reproducible.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Sequence, Tuple

import torch

from .config import AugConfig

_LUMA = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)


def _ensure_4d(tensor: torch.Tensor) -> tuple[torch.Tensor, callable]:
    if tensor is None:  # pragma: no cover - guard for optional inputs
        return tensor, lambda x: x
    batch_added = False
    channel_added = False
    out = tensor
    if out.dim() == 2:
        out = out.unsqueeze(0).unsqueeze(0)
        batch_added = True
        channel_added = True
    elif out.dim() == 3:
        out = out.unsqueeze(0)
        batch_added = True
    elif out.dim() != 4:
        raise ValueError("Expected tensor with 2D/3D/4D shape")

    def _restore(t: torch.Tensor) -> torch.Tensor:
        restored = t
        if batch_added:
            restored = restored.squeeze(0)
        if channel_added:
            restored = restored.squeeze(0)
        return restored

    return out, _restore


def _validate_rotations(choices: Sequence[int]) -> Sequence[int]:
    normalized = [int(c) for c in (choices or [])]
    for angle in normalized:
        if angle % 90 != 0:
            raise ValueError("rotate_choices must be multiples of 90 degrees")
    return normalized


def _sample_choice(values: Sequence[int], rng: Optional[torch.Generator]) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return int(values[0])
    idx = int(torch.randint(len(values), (1,), generator=rng).item())
    return int(values[idx])


def _sample_bool(prob: float, rng: Optional[torch.Generator]) -> bool:
    prob = float(max(0.0, min(1.0, prob)))
    return bool(torch.rand((), generator=rng).item() < prob)


def _sample_range(bounds: Tuple[float, float], rng: Optional[torch.Generator]) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi < lo:
        raise ValueError("Range upper bound must be >= lower bound")
    if hi == lo:
        return lo
    return float(lo + (hi - lo) * torch.rand((), generator=rng).item())


def _apply_geometry(tensor: torch.Tensor, *, turns: int, flip_h: bool, flip_v: bool) -> torch.Tensor:
    out = tensor
    if flip_h:
        out = torch.flip(out, dims=[-1])
    if flip_v:
        out = torch.flip(out, dims=[-2])
    if turns:
        out = torch.rot90(out, turns, dims=(-2, -1))
    return out


def _apply_photometric(rgb: torch.Tensor, cfg: AugConfig, rng: Optional[torch.Generator]) -> torch.Tensor:
    out = rgb.float()
    if not _sample_bool(cfg.photometric_prob, rng):
        return out

    noise_bounds = getattr(cfg, "noise_std_range", None)
    if noise_bounds is None:
        sigma = float(max(0.0, getattr(cfg, "gaussian_noise_std", 0.0)))
        noise_bounds = (sigma, sigma)
    sigma = _sample_range(noise_bounds, rng)
    if sigma > 0:
        if rng is None:
            noise = torch.randn_like(out) * sigma
        else:
            noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=rng) * sigma
        out = (out + noise).clamp(0.0, 1.0)

    contrast = _sample_range(getattr(cfg, "contrast_range", (1.0, 1.0)), rng)
    if contrast != 1.0:
        mean = out.mean(dim=(-2, -1), keepdim=True)
        out = ((out - mean) * contrast + mean).clamp(0.0, 1.0)

    saturation = _sample_range(getattr(cfg, "saturation_range", (1.0, 1.0)), rng)
    if saturation != 1.0:
        weights = _LUMA.to(device=out.device, dtype=out.dtype).view(1, 3, 1, 1)
        gray = (out * weights).sum(dim=1, keepdim=True)
        out = (gray + saturation * (out - gray)).clamp(0.0, 1.0)

    return out


def make_rng(
    base_seed: Optional[int],
    *,
    worker_id: int = 0,
    sample_index: int = 0,
    item_id: Optional[object] = None,
    view_id: int = 0,
) -> Optional[torch.Generator]:
    """Build a deterministic torch.Generator from a base seed and identifiers.

    Args:
        base_seed: global augmentation seed (None disables determinism).
        worker_id: DataLoader worker id (0 for main process).
        sample_index: local sample counter (monotonic per dataset iterator).
        item_id: stable item identifier (e.g., shard item_id) used to avoid
            reordering drift across workers.
        view_id: view selector (use different values for view1/view2).
    """

    if base_seed is None:
        return None
    h = hashlib.sha256()
    for token in (int(base_seed), int(worker_id), int(sample_index), int(view_id)):
        h.update(str(token).encode("utf-8"))
    if item_id is not None:
        h.update(str(item_id).encode("utf-8"))
    seed_int = int.from_bytes(h.digest()[:8], byteorder="little", signed=False)
    generator = torch.Generator()
    generator.manual_seed(seed_int)
    return generator


def apply_augmentations(
    rgb: torch.Tensor,
    *,
    slic: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    aug_cfg: Optional[AugConfig] = None,
    rng: Optional[torch.Generator] = None,
) -> dict:
    """Apply synchronized geometry + photometric transforms to a sample.

    Args:
        rgb: Tensor shaped [C,H,W] or [B,C,H,W] in [0,1].
        slic: Optional label map aligned to rgb (int tensor, [H,W]/[B,H,W]/[B,1,H,W]).
        target: Optional aligned label map treated like slic for geometry only.
        aug_cfg: AugConfig with probabilities and ranges.
        rng: Optional torch.Generator for deterministic sampling.
    Returns:
        Dict with keys ``rgb`` (float tensor), ``slic`` (same dtype as input),
        and ``target`` (same dtype as input) where provided.
    """

    cfg = aug_cfg or AugConfig()
    if not getattr(cfg, "enabled", True):
        return {"rgb": rgb, "slic": slic, "target": target}

    rotations = _validate_rotations(getattr(cfg, "rotate_choices", (0,)))

    rgb_4d, restore_rgb = _ensure_4d(rgb)
    if rgb_4d.shape[1] != 3:
        raise ValueError("RGB tensor must have 3 channels")
    slic_4d, restore_slic = _ensure_4d(slic) if slic is not None else (None, None)
    target_4d, restore_target = _ensure_4d(target) if target is not None else (None, None)

    turns = (_sample_choice(rotations, rng) // 90) % 4 if rotations else 0
    flip_h = _sample_bool(getattr(cfg, "flip_h_prob", cfg.flip_prob), rng)
    flip_v = _sample_bool(getattr(cfg, "flip_v_prob", 0.0), rng)

    rgb_geo = _apply_geometry(rgb_4d, turns=turns, flip_h=flip_h, flip_v=flip_v)
    slic_geo = _apply_geometry(slic_4d, turns=turns, flip_h=flip_h, flip_v=flip_v) if slic_4d is not None else None
    target_geo = _apply_geometry(target_4d, turns=turns, flip_h=flip_h, flip_v=flip_v) if target_4d is not None else None

    rgb_out = _apply_photometric(rgb_geo, cfg, rng)

    result = {"rgb": restore_rgb(rgb_out)}
    if slic_geo is not None and restore_slic is not None:
        result["slic"] = restore_slic(slic_geo).to(slic.dtype)
    else:
        result["slic"] = None
    if target_geo is not None and restore_target is not None:
        result["target"] = restore_target(target_geo).to(target.dtype)
    else:
        result["target"] = target
    return result


__all__ = ["apply_augmentations", "make_rng"]
