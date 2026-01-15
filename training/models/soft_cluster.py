# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Differentiable soft k-means head."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SoftKMeansHead(nn.Module):
    def __init__(self, embed_dim: int, max_k: int = 16):
        super().__init__()
        self.max_k = max_k
        self.seed_proj = nn.Conv2d(embed_dim, max_k, kernel_size=1)

    def forward(
        self,
        embeddings: torch.Tensor,
        k: int,
        cluster_iters: int = 3,
        temperature: float = 0.8,
        stop_grad_prototypes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute soft assignments and prototypes.

        Returns probs [B,K,512,512], prototypes [B,K,C], logits at latent stride.
        """
        if k < 2 or k > self.max_k:
            raise ValueError(f"k must be in [2,{self.max_k}] (got {k})")
        b, c, h, w = embeddings.shape
        cluster_iters = max(1, int(cluster_iters))
        temp = max(1e-4, float(temperature))

        seeds = self.seed_proj(embeddings)[:, :k]
        assign = torch.softmax(seeds.view(b, k, -1), dim=1)
        feats = embeddings.view(b, c, -1)
        prototypes = torch.einsum("bkn,bcn->bkc", assign, feats) / (assign.sum(dim=2, keepdim=False).unsqueeze(-1) + 1e-6)

        for _ in range(cluster_iters):
            if stop_grad_prototypes:
                proto = prototypes.detach()
            else:
                proto = prototypes
            proto_expand = proto.view(b, k, c, 1)
            feat_expand = feats.view(b, 1, c, h * w)
            dist = (feat_expand - proto_expand).pow(2).sum(dim=2)  # [B,K,N]
            logits = -dist / temp
            assign = torch.softmax(logits, dim=1)
            denom = assign.sum(dim=2, keepdim=True) + 1e-6
            prototypes = torch.einsum("bkn,bcn->bkc", assign, feats) / denom

        logits_map = logits.view(b, k, h, w)
        probs = torch.softmax(logits_map, dim=1)
        probs_full = F.interpolate(probs, scale_factor=4, mode="bilinear", align_corners=False)
        return probs_full, prototypes, logits_map
