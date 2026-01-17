# SPDX-License-Identifier: BSD-3-Clause
"""Lightweight CNN student producing stride-4 embeddings for variable-K clustering."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _Residual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class StudentEmbeddingNet(nn.Module):
    """CNN-only student embedding model.

    Constraints: stride=4, <=10M params for embed_dim<=192.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        width = max(32, min(256, embed_dim))
        self.stem = nn.Sequential(
            nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.block1 = _Residual(width)
        self.block2 = _Residual(width)
        self.proj = nn.Conv2d(width, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError("Expected rgb BCHW with 3 channels")
        feat = self.stem(x)
        feat = self.block1(feat)
        feat = self.block2(feat)
        emb = self.proj(feat)
        return emb  # [B, D, H/4, W/4]

    @staticmethod
    def param_count(model: "StudentEmbeddingNet") -> int:
        return sum(p.numel() for p in model.parameters())


def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def batched_kmeans(emb: torch.Tensor, k: int, iters: int = 5, temperature: float = 0.8):
    """Simplified batched k-means on embeddings (B,C,H,W -> assignments, prototypes)."""
    b, c, h, w = emb.shape
    flat = emb.view(b, c, -1)
    n = flat.shape[-1]
    k_eff = max(1, min(k, n))
    # init by slicing evenly spaced points
    step = max(1, n // k_eff)
    proto = flat[:, :, ::step][:, :, :k_eff].clone().permute(0, 2, 1)  # [B, K, C]
    for _ in range(max(1, iters)):
        diff = flat.unsqueeze(1) - proto.unsqueeze(-1)  # [B, K, C, N]
        dist = (diff * diff).sum(dim=2)  # [B, K, N]
        logits = -dist / max(temperature, 1e-6)
        assign = torch.softmax(logits, dim=1)
        denom = assign.sum(dim=2, keepdim=True) + 1e-6
        proto = torch.einsum("bkn,bcn->bkc", assign, flat) / denom
    assign_map = assign.view(b, k_eff, h, w)
    return assign_map, proto


def affinity_matrix(emb: torch.Tensor, sample: int = 512):
    b, c, h, w = emb.shape
    flat = emb.view(b, c, -1)
    n = flat.shape[-1]
    sample = min(sample, n)
    idx = torch.randperm(n, device=emb.device)[:sample]
    sub = flat[:, :, idx]
    sub = l2_normalize(sub, dim=1)
    sim = torch.einsum("bci,bcj->bij", sub, sub)
    return sim
