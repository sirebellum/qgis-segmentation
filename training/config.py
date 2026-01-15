# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Lightweight configuration dataclasses for unsupervised training scaffolding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class ModelConfig:
    embed_dim: int = 96
    max_k: int = 16
    temperature: float = 0.8
    proto_momentum: float = 0.0  # 0 means standard soft k-means updates
    cluster_iters: Tuple[int, int] = (2, 6)
    smoothing_iters: Tuple[int, int] = (1, 3)
    smoothing_lanes: Sequence[str] = ("fast", "learned", "none")
    downsample_choices: Sequence[int] = (1, 2)
    elev_film_channels: int = 32
    elev_dropout: float = 0.5
    stop_grad_prototypes: bool = False


@dataclass
class DataConfig:
    patch_size: int = 512
    stride: int = 512
    elevation_dropout: float = 0.3
    allow_mixed_elevation: bool = True
    backend: str = "gdal"  # placeholder hint; dataset handles availability


@dataclass
class AugConfig:
    flip_prob: float = 0.5
    rotate_choices: Sequence[int] = (0, 90, 180, 270)
    color_jitter: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.05)
    gaussian_noise_std: float = 0.01
    max_affine_deg: float = 10.0


@dataclass
class LossConfig:
    consistency_weight: float = 1.0
    entropy_min_weight: float = 0.2
    entropy_marginal_weight: float = 0.6
    smoothness_weight: float = 0.3
    edge_weight: float = 0.5


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    steps: int = 1000
    batch_size: int = 2
    grad_accum: int = 1
    amp: bool = True
    log_interval: int = 20
    checkpoint_path: Optional[str] = None
    seed: int = 42


@dataclass
class KnobConfig:
    k_choices: Sequence[int] = (2, 4, 8, 16)
    cluster_iters_range: Tuple[int, int] = (2, 6)
    smooth_iters_range: Tuple[int, int] = (1, 3)
    downsample_choices: Sequence[int] = (1, 2)
    smoothing_lanes: Sequence[str] = ("fast", "learned", "none")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    knobs: KnobConfig = field(default_factory=KnobConfig)
    preset_name: str = "default"


def default_config() -> Config:
    """Return the default config instance."""
    return Config()
