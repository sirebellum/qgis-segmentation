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
    stop_grad_prototypes: bool = False


@dataclass
class DataConfig:
    patch_sizes: Sequence[int] = (256, 512, 1024)
    patch_size_sampling: str = "uniform"  # uniform | weighted (not yet implemented)
    patch_size: int = 512
    stride: int = 512
    backend: str = "gdal"  # placeholder hint; dataset handles availability
    manifest_path: Optional[str] = None
    max_samples: Optional[int] = None
    source: str = "synthetic"  # synthetic | shards
    processed_root: str = "training/datasets/processed"
    dataset_id: Optional[str] = None
    train_split: str = "train"
    metrics_split: str = "metrics_train"
    val_split: str = "val"
    cache_mode: str = "none"  # none | lru
    cache_max_items: int = 0
    require_slic: bool = True
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    pin_memory: bool = False
    iou_ignore_label_leq: int = 0
    raster_paths: Optional[List[str]] = None  # real GeoTIFF inputs (3-band)
    target_paths: Optional[List[str]] = None  # aligned target rasters (optional)


@dataclass
class AugConfig:
    enabled: bool = True
    seed: Optional[int] = None
    flip_prob: float = 0.5  # legacy alias for horizontal flips
    flip_h_prob: float = 0.5
    flip_v_prob: float = 0.0
    rotate_choices: Sequence[int] = (0, 90, 180, 270)
    photometric_prob: float = 1.0
    noise_std_range: Tuple[float, float] = (0.0, 0.01)
    contrast_range: Tuple[float, float] = (0.9, 1.1)
    saturation_range: Tuple[float, float] = (0.9, 1.1)
    color_jitter: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.05)  # deprecated
    gaussian_noise_std: float = 0.01  # deprecated; superseded by noise_std_range
    max_affine_deg: float = 10.0  # placeholder for future affine aug


@dataclass
class LossConfig:
    consistency_weight: float = 1.0
    entropy_min_weight: float = 0.2
    entropy_marginal_weight: float = 0.6
    smoothness_weight: float = 0.3
    edge_weight: float = 0.5
    slic_boundary_weight: float = 0.0
    slic_antimerge_weight: float = 0.0
    slic_within_weight: float = 0.0


@dataclass
class AutoencoderLossConfig:
    """Training-only autoencoder reconstruction loss (Option C + D).
    
    This loss encourages structure/blob fidelity via:
    - Low-pass RGB reconstruction at stride-4 (Option C)
    - Gradient/edge consistency at stride-4 (Option D)
    
    The decoder is training-only and excluded from deployment artifacts.
    """
    enabled: bool = True  # on by default
    lambda_recon: float = 0.01  # small coefficient to avoid dominating
    ema_decay: float = 0.99  # EMA decay for normalization
    blur_sigma: float = 1.0  # Gaussian blur sigma for low-pass target
    blur_kernel: int = 5  # Gaussian kernel size
    grad_weight: float = 0.2  # relative weight for gradient vs blur loss
    detach_backbone: bool = False  # if True, detach features from backbone grad
    hidden_channels: int = 64  # decoder hidden channels
    num_blocks: int = 2  # number of decoder conv blocks


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    steps: int = 1000
    batch_size: int = 2
    grad_accum: int = 1
    amp: bool = True
    log_interval: int = 1
    log_image_interval: int = 1
    steps_per_epoch: Optional[int] = None
    evaluation_epoch: int = 1
    checkpoint_path: Optional[str] = None
    seed: int = 42
    eval_interval: int = 0
    multi_scale_mode: str = "sample_one_scale_per_step"  # sample_one_scale_per_step | per_step_all_scales


@dataclass
class TeacherConfig:
    enabled: bool = False
    name: str = "fake"
    proj_dim: int = 256
    feature_weight: float = 1.0
    affinity_weight: float = 1.0
    sample: int = 256
    patch_mode: str = "match"  # match | high | medium | low


@dataclass
class StudentConfig:
    enabled: bool = True
    embed_dim: int = 96
    depth: int = 3
    norm: str = "group"  # group | batch | instance | none
    groups: int = 8
    dropout: float = 0.0


@dataclass
class DistillLossConfig:
    feature_weight: float = 1.0
    affinity_weight: float = 1.0
    affinity_sample: int = 256
    clustering_weight: float = 1.0
    cluster_iters: int = 3
    temperature: float = 0.8
    tv_weight: float = 0.1
    consistency_weight: float = 0.3
    ema_decay: float = 0.9
    merge_eps: float = 1e-4
    consistency_ema_decay: float = 0.9
    consistency_eps: float = 1e-4
    boundary_weight: float = 1.0
    antimerge_weight: float = 1.0
    within_weight: float = 0.05


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
    autoencoder: AutoencoderLossConfig = field(default_factory=AutoencoderLossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    knobs: KnobConfig = field(default_factory=KnobConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    distill: DistillLossConfig = field(default_factory=DistillLossConfig)
    preset_name: str = "default"


def default_config() -> Config:
    """Return the default config instance."""
    return Config()
