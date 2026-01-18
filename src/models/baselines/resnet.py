"""ResNet-1D 베이스라인 모델.

Reference: Wang et al., "Time Series Classification from Scratch" (2017)
https://arxiv.org/abs/1611.06455
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """1D Residual Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 8,
        stride: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5,
            padding=2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels)

        # Shortcut
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + identity
        out = F.relu(out)

        return out


class ResNet1D(nn.Module):
    """1D ResNet for time series classification."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_dims: List[int] = [64, 128, 128],
        dropout: float = 0.0,
    ):
        """
        Args:
            input_size: 시계열 길이
            num_classes: 클래스 수
            hidden_dims: 각 ResBlock의 채널 수
            dropout: Dropout 비율
        """
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # Residual Blocks
        blocks = []
        in_channels = 1
        for out_channels in hidden_dims:
            blocks.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

        # Global Average Pooling + Classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) -> (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.blocks(x)
        x = self.gap(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
