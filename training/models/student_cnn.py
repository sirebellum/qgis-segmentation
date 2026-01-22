# SPDX-License-Identifier: BSD-3-Clause
"""CNN student that is input-size invariant (stride-4 embeddings)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _norm_layer(channels: int, norm: str, groups: int) -> nn.Module:
    norm = (norm or "none").lower()
    if norm == "group":
        groups_eff = max(1, min(groups, channels))
        while channels % groups_eff != 0 and groups_eff > 1:
            groups_eff -= 1
        return nn.GroupNorm(groups_eff, channels)
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    return nn.Identity()


class VGGBlock3x3(nn.Module):
    """Resolution-preserving stack of 3x3 convs with norm/activation."""

    def __init__(self, channels: int, depth: int, norm: str = "group", groups: int = 8, dropout: float = 0.0):
        super().__init__()
        layers = []
        for _ in range(max(1, depth)):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            layers.append(_norm_layer(channels, norm, groups))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentEmbeddingNet(nn.Module):
    """CNN-only student producing a single stride-4 embedding map.

    The network is fully convolutional and accepts any H/W divisible by 4.
    Patch-size differences (e.g., 256/512/1024) only change the output grid
    shape; the weights are shared across all scales.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 96,
        depth: int = 3,
        norm: str = "group",
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        self.embed_dim = embed_dim
        self.depth = max(1, depth)
        self.norm = norm
        self.groups = groups
        self.dropout = dropout
        base = max(64, embed_dim)
        self.pre_proj_channels = base  # exposed for decoder

        # Two stride-2 stems to reach stride 4.
        self.stem = nn.Sequential(
            nn.Conv2d(3, base // 2, kernel_size=3, stride=2, padding=1),
            _norm_layer(base // 2, norm, groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, base, kernel_size=3, stride=2, padding=1),
            _norm_layer(base, norm, groups),
            nn.ReLU(inplace=True),
        )

        # Depth-wise VGG blocks at stride 4.
        self.body = VGGBlock3x3(base, depth=self.depth, norm=norm, groups=groups, dropout=dropout)

        # Projection + optional refinement.
        self.head = nn.Sequential(
            nn.Conv2d(base, base, kernel_size=3, stride=1, padding=1),
            _norm_layer(base, norm, groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, embed_dim, kernel_size=1),
        )
        self.refine = VGGBlock3x3(embed_dim, depth=1, norm=norm, groups=groups, dropout=dropout)
        self.output_stride = 4

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        """Forward pass with optional pre-projection feature return.
        
        Args:
            x: RGB input [B,3,H,W]
            return_features: if True, return (embeddings, pre_proj_features)
                for use by training-only reconstruction decoder
                
        Returns:
            embeddings [B,embed_dim,H/4,W/4] if return_features=False
            (embeddings, features) tuple if return_features=True
        """
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError("Expected rgb BCHW with 3 channels")
        if x.shape[-1] % self.output_stride != 0 or x.shape[-2] % self.output_stride != 0:
            raise ValueError(f"H and W must be divisible by {self.output_stride} for the student")

        feat = self.stem(x)
        feat = self.body(feat)
        # feat is now the pre-projection feature map at stride-4
        emb = self.head(feat)
        emb = self.refine(emb)
        
        if return_features:
            return emb, feat
        return emb

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
    step = max(1, n // k_eff)
    proto = flat[:, :, ::step][:, :, :k_eff].clone().permute(0, 2, 1)  # [B, K, C]
    for _ in range(max(1, iters)):
        diff = flat.unsqueeze(1) - proto.unsqueeze(-1)
        dist = (diff * diff).sum(dim=2)
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
