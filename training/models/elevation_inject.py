# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Gated FiLM-style elevation injection into latent embeddings."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.resample import resample_to_match


class ElevationFiLM(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, embed_dim * 2, 1),
        )

    def forward(self, embeddings: torch.Tensor, elev: torch.Tensor, elev_present) -> torch.Tensor:
        if elev is None:
            return embeddings

        mask = None
        if isinstance(elev_present, torch.Tensor):
            mask = elev_present.to(embeddings.device)
            if mask.dim() == 1:
                mask = mask.view(-1, 1, 1, 1)
            elif mask.dim() == 4:
                pass
            else:
                raise ValueError("elev_present mask must be 1D batch or broadcastable")
        elif not elev_present:
            return embeddings

        elev_ds = resample_to_match(elev, embeddings.shape)
        film = self.mlp(elev_ds)
        gamma, beta = torch.chunk(film, 2, dim=1)
        if mask is not None:
            gamma = gamma * mask
            beta = beta * mask
        return embeddings * (1 + gamma) + beta
