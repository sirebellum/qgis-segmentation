# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Monolithic unsupervised segmentation model (training-only scaffolding)."""
from __future__ import annotations

import random
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from ..utils.resample import downsample_factor, resample_to_match
from .backbone import Encoder
from .elevation_inject import ElevationFiLM
from .refine import RefineHead, fast_smooth
from .soft_cluster import SoftKMeansHead


class MonolithicSegmenter(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(embed_dim=cfg.embed_dim)
        self.elev_gate = ElevationFiLM(embed_dim=cfg.embed_dim, hidden=cfg.elev_film_channels)
        self.cluster_head = SoftKMeansHead(embed_dim=cfg.embed_dim, max_k=cfg.max_k)
        self.refiner = RefineHead(num_classes=cfg.max_k)

    def forward(
        self,
        rgb: torch.Tensor,
        k: int,
        elev: Optional[torch.Tensor] = None,
        elev_present: Optional[bool] = None,
        downsample: int = 1,
        cluster_iters: Optional[int] = None,
        smooth_iters: Optional[int] = None,
        smoothing_lane: str = "fast",
    ) -> dict:
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must be [B,3,H,W]")
        if k < 2 or k > self.cfg.max_k:
            raise ValueError(f"k must be within [2,{self.cfg.max_k}] (got {k})")
        ds = max(1, int(downsample))
        rgb_ds = downsample_factor(rgb, ds)
        emb = self.encoder(rgb_ds)

        elev_present_flag = bool(elev_present) if elev_present is not None else elev is not None
        if elev is not None and elev_present_flag:
            elev = resample_to_match(elev, rgb_ds.shape)
            emb = self.elev_gate(emb, elev, elev_present_flag)

        cluster_iters = int(cluster_iters) if cluster_iters is not None else int(
            sum(self.cfg.cluster_iters) / 2
        )
        smooth_iters = int(smooth_iters) if smooth_iters is not None else int(
            sum(self.cfg.smoothing_iters) / 2
        )

        probs_full, prototypes, logits_latent = self.cluster_head(
            emb,
            k,
            cluster_iters=cluster_iters,
            temperature=self.cfg.temperature,
            stop_grad_prototypes=self.cfg.stop_grad_prototypes,
        )

        # Upsample to original spatial size
        probs = F.interpolate(probs_full, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        if smoothing_lane == "learned":
            pad = self.cfg.max_k - k
            if pad > 0:
                probs_in = torch.cat([probs, torch.zeros(probs.size(0), pad, *probs.shape[-2:], device=probs.device)], dim=1)
            else:
                probs_in = probs
            refined = self.refiner(probs_in, rgb)
            probs = refined[:, :k]
        elif smoothing_lane == "fast":
            probs = fast_smooth(probs, iters=smooth_iters)

        logits = torch.log(probs.clamp(min=1e-6))
        return {
            "probs": probs,
            "logits": logits,
            "embeddings": emb,
            "prototypes": prototypes,
            "logits_latent": logits_latent,
        }

    def sample_knobs(self) -> dict:
        cluster_iters = random.randint(self.cfg.cluster_iters[0], self.cfg.cluster_iters[1])
        smooth_iters = random.randint(self.cfg.smoothing_iters[0], self.cfg.smoothing_iters[1])
        downsample = random.choice(self.cfg.downsample_choices)
        smoothing_lane = random.choice(self.cfg.smoothing_lanes)
        k = random.choice(range(2, self.cfg.max_k + 1, 2))
        return {
            "cluster_iters": cluster_iters,
            "smooth_iters": smooth_iters,
            "downsample": downsample,
            "smoothing_lane": smoothing_lane,
            "k": k,
        }
