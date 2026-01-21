# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Adaptive chunking and batching settings for the runtime pipeline."""
from __future__ import annotations

import math
import os
import platform
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .common import _coerce_torch_device

try:  # Optional dependency for better memory stats
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

DEFAULT_MEMORY_BUDGET = 128 * 1024 * 1024
VRAM_RATIO_CUDA = 0.009
VRAM_RATIO_MPS = 0.0075
VRAM_RATIO_CPU = 0.01
MIN_TILE_SIZE = 128
MAX_TILE_SIZE = 512


@dataclass
class ChunkPlan:
    chunk_size: int
    overlap: int
    budget_bytes: int
    ratio: float
    prefetch_depth: int

    def __post_init__(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})")

    @property
    def stride(self) -> int:
        return max(1, self.chunk_size - self.overlap)

    def should_chunk(self, height: int, width: int) -> bool:
        return height > self.chunk_size or width > self.chunk_size


@dataclass
class AdaptiveSettings:
    safety_factor: int = 8
    prefetch_depth: int = 2


def _copy_setting(settings: AdaptiveSettings) -> AdaptiveSettings:
    return AdaptiveSettings(safety_factor=settings.safety_factor, prefetch_depth=settings.prefetch_depth)


_ADAPTIVE_SETTINGS_MAP: Dict[str, AdaptiveSettings] = {"default": AdaptiveSettings()}
_ADAPTIVE_OPTIONS_MAP: Dict[str, List[Tuple[int, AdaptiveSettings]]] = {"default": []}
_ADAPTIVE_DEFAULT_TIER = "default"


def get_adaptive_settings(memory_bytes: Optional[int] = None, tier: Optional[str] = None) -> AdaptiveSettings:
    tier_key = tier or _ADAPTIVE_DEFAULT_TIER
    settings = _ADAPTIVE_SETTINGS_MAP.get(tier_key) or _ADAPTIVE_SETTINGS_MAP.get(_ADAPTIVE_DEFAULT_TIER)
    if settings is None:
        settings = AdaptiveSettings()
    options = _ADAPTIVE_OPTIONS_MAP.get(tier_key, [])
    if memory_bytes is not None and options:
        for threshold, opt_settings in reversed(options):
            if memory_bytes >= threshold:
                return opt_settings
        return options[0][1]
    return settings


def set_adaptive_settings(
    settings: Optional[Dict[str, AdaptiveSettings]] = None,
    options: Optional[Dict[str, List[Tuple[int, AdaptiveSettings]]]] = None,
    default_tier: Optional[str] = None,
) -> None:
    global _ADAPTIVE_SETTINGS_MAP, _ADAPTIVE_OPTIONS_MAP, _ADAPTIVE_DEFAULT_TIER

    if settings is None:
        settings = {"default": AdaptiveSettings()}
    elif not isinstance(settings, dict):  # backward compatibility
        settings = {default_tier or "default": settings}

    _ADAPTIVE_SETTINGS_MAP = {tier: _copy_setting(cfg) for tier, cfg in settings.items()}

    if options is None:
        _ADAPTIVE_OPTIONS_MAP = {tier: [] for tier in _ADAPTIVE_SETTINGS_MAP}
    else:
        normalized: Dict[str, List[Tuple[int, AdaptiveSettings]]] = {}
        for tier, entries in options.items():
            sorted_entries = sorted(entries, key=lambda item: item[0])
            normalized[tier] = [
                (int(max(0, threshold)), _copy_setting(cfg))
                for threshold, cfg in sorted_entries
            ]
        for tier in _ADAPTIVE_SETTINGS_MAP:
            normalized.setdefault(tier, [])
        _ADAPTIVE_OPTIONS_MAP = normalized

    if default_tier and default_tier in _ADAPTIVE_SETTINGS_MAP:
        _ADAPTIVE_DEFAULT_TIER = default_tier
    elif _ADAPTIVE_DEFAULT_TIER not in _ADAPTIVE_SETTINGS_MAP:
        _ADAPTIVE_DEFAULT_TIER = next(iter(_ADAPTIVE_SETTINGS_MAP))


def get_adaptive_options(tier: Optional[str] = None) -> List[Tuple[int, AdaptiveSettings]]:
    tier_key = tier or _ADAPTIVE_DEFAULT_TIER
    return list(_ADAPTIVE_OPTIONS_MAP.get(tier_key, []))


def _free_vram_bytes(device):
    if device.type == "cuda" and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(device.index or torch.cuda.current_device())
        return free
    return _system_available_memory()


def _system_available_memory():
    if psutil is not None:
        return int(psutil.virtual_memory().available)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * avail_pages)
    except (ValueError, AttributeError, OSError):  # pragma: no cover
        return 1_000_000_000


def _derive_chunk_size(array_shape, device, profile_tier: Optional[str] = None):
    channels = array_shape[0]
    free_bytes = _free_vram_bytes(device)
    if device.type == "cuda":
        ratio = VRAM_RATIO_CUDA
    elif device.type == "mps":
        ratio = VRAM_RATIO_MPS
    else:
        ratio = VRAM_RATIO_CPU
    budget = max(int(free_bytes * ratio), 64 * 1024 * 1024)
    bytes_per_pixel = channels * 4
    settings = get_adaptive_settings(free_bytes, tier=profile_tier)
    safety = settings.safety_factor
    max_pixels = max(budget // (bytes_per_pixel * safety), 1)
    tile_side = int(math.sqrt(max_pixels))
    tile_side = max(MIN_TILE_SIZE, min(MAX_TILE_SIZE, tile_side))
    return tile_side, budget, ratio, settings


def recommended_chunk_plan(array_shape, device, profile_tier: Optional[str] = None) -> ChunkPlan:
    chunk_size, budget, ratio, settings = _derive_chunk_size(array_shape, device, profile_tier=profile_tier)
    overlap = 0
    return ChunkPlan(
        chunk_size=chunk_size,
        overlap=overlap,
        budget_bytes=budget,
        ratio=ratio,
        prefetch_depth=settings.prefetch_depth,
    )


__all__ = [
    "ChunkPlan",
    "AdaptiveSettings",
    "DEFAULT_MEMORY_BUDGET",
    "VRAM_RATIO_CPU",
    "VRAM_RATIO_CUDA",
    "VRAM_RATIO_MPS",
    "MIN_TILE_SIZE",
    "MAX_TILE_SIZE",
    "get_adaptive_settings",
    "set_adaptive_settings",
    "get_adaptive_options",
    "recommended_chunk_plan",
    "_derive_chunk_size",
    "_system_available_memory",
    "_free_vram_bytes",
]
