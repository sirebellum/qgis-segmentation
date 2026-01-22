# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Tests for runtime/adaptive.py - adaptive settings and chunk planning."""

import numpy as np
import pytest
import torch

from runtime.adaptive import (
    AdaptiveSettings,
    ChunkPlan,
    DEFAULT_MEMORY_BUDGET,
    MAX_TILE_SIZE,
    MIN_TILE_SIZE,
    VRAM_RATIO_CPU,
    VRAM_RATIO_CUDA,
    VRAM_RATIO_MPS,
    _derive_chunk_size,
    _free_vram_bytes,
    _system_available_memory,
    get_adaptive_options,
    get_adaptive_settings,
    recommended_chunk_plan,
    set_adaptive_settings,
)


class TestAdaptiveSettings:
    def test_defaults(self):
        settings = AdaptiveSettings()
        assert settings.safety_factor > 0
        assert settings.prefetch_depth >= 0

    def test_set_and_get(self):
        custom = AdaptiveSettings(safety_factor=10, prefetch_depth=3)
        set_adaptive_settings({"default": custom})
        retrieved = get_adaptive_settings()
        assert retrieved.safety_factor == 10
        assert retrieved.prefetch_depth == 3
        # Reset to default
        set_adaptive_settings({"default": AdaptiveSettings()})

    def test_tiered_settings(self):
        tier_map = {
            "low": AdaptiveSettings(safety_factor=4, prefetch_depth=1),
            "high": AdaptiveSettings(safety_factor=16, prefetch_depth=4),
        }
        set_adaptive_settings(tier_map, default_tier="low")
        low = get_adaptive_settings()
        assert low.safety_factor == 4
        # Reset
        set_adaptive_settings({"default": AdaptiveSettings()})


class TestChunkPlan:
    def test_chunk_plan_fields(self):
        plan = ChunkPlan(
            chunk_size=64,
            overlap=8,
            budget_bytes=1024 * 1024,
            ratio=0.5,
            prefetch_depth=2,
        )
        assert plan.chunk_size == 64
        assert plan.overlap == 8
        assert plan.budget_bytes == 1024 * 1024
        assert plan.ratio == 0.5
        assert plan.prefetch_depth == 2

    def test_stride_property(self):
        plan = ChunkPlan(chunk_size=64, overlap=16, budget_bytes=1024, ratio=0.5, prefetch_depth=1)
        assert plan.stride == 48

    def test_should_chunk_large_input(self):
        plan = ChunkPlan(chunk_size=64, overlap=0, budget_bytes=1024, ratio=0.5, prefetch_depth=1)
        assert plan.should_chunk(128, 128) is True

    def test_should_not_chunk_small_input(self):
        plan = ChunkPlan(chunk_size=64, overlap=0, budget_bytes=1024, ratio=0.5, prefetch_depth=1)
        assert plan.should_chunk(32, 32) is False


class TestDeriveChunkSize:
    def test_returns_tuple(self):
        result = _derive_chunk_size((3, 64, 64), torch.device("cpu"))
        assert isinstance(result, tuple)
        assert len(result) == 4
        chunk_size, budget, ratio, settings = result
        assert chunk_size >= MIN_TILE_SIZE
        assert chunk_size <= MAX_TILE_SIZE

    def test_respects_bounds(self):
        result = _derive_chunk_size((3, 1024, 1024), torch.device("cpu"))
        chunk_size = result[0]
        assert MIN_TILE_SIZE <= chunk_size <= MAX_TILE_SIZE


class TestRecommendedChunkPlan:
    def test_returns_valid_plan(self):
        plan = recommended_chunk_plan((3, 256, 256), torch.device("cpu"))
        assert isinstance(plan, ChunkPlan)
        assert plan.chunk_size >= MIN_TILE_SIZE
        assert plan.chunk_size <= MAX_TILE_SIZE

    def test_respects_device_hint(self):
        cpu_plan = recommended_chunk_plan((3, 256, 256), torch.device("cpu"))
        assert cpu_plan.chunk_size >= MIN_TILE_SIZE


class TestMemoryProbes:
    def test_system_memory_positive(self):
        mem = _system_available_memory()
        assert mem > 0

    def test_free_vram_cpu_fallback(self):
        result = _free_vram_bytes(torch.device("cpu"))
        assert result > 0


class TestVRAMRatios:
    def test_ratios_are_sane(self):
        assert 0 < VRAM_RATIO_CPU <= 1
        assert 0 < VRAM_RATIO_CUDA <= 1
        assert 0 < VRAM_RATIO_MPS <= 1


class TestAdaptiveOptions:
    def test_get_adaptive_options_returns_list(self):
        opts = get_adaptive_options()
        assert isinstance(opts, list)
