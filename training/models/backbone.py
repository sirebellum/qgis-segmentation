# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Small CNN backbone emitting stride-4 embeddings."""
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class Encoder(nn.Module):
    def __init__(self, embed_dim: int = 96):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResidualBlock(embed_dim)
        self.block2 = ResidualBlock(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 4 != 0 or x.shape[-1] % 4 != 0:
            raise ValueError("Encoder expects H and W divisible by 4")
        out = self.stem(x)
        out = self.block1(out)
        out = self.block2(out)
        return out  # [B, D, H/4, W/4]
