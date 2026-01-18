"""FCN (Fully Convolutional Network) 베이스라인 모델.

Reference: Wang et al., "Time Series Classification from Scratch" (2017)
https://arxiv.org/abs/1611.06455
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    """Fully Convolutional Network for time series classification.

    간단한 3-layer CNN with Global Average Pooling.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_dims: List[int] = [128, 256, 128],
        kernel_sizes: List[int] = [8, 5, 3],
        dropout: float = 0.0,
    ):
        """
        Args:
            input_size: 시계열 길이
            num_classes: 클래스 수
            hidden_dims: 각 Conv 레이어의 채널 수
            kernel_sizes: 각 Conv 레이어의 커널 크기
            dropout: Dropout 비율
        """
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        assert len(hidden_dims) == len(kernel_sizes), \
            "hidden_dims와 kernel_sizes의 길이가 같아야 합니다."

        # Convolutional layers
        layers = []
        in_channels = 1

        for out_channels, kernel_size in zip(hidden_dims, kernel_sizes):
            layers.extend([
                nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    padding=kernel_size // 2, bias=False
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Global Average Pooling + Classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) -> (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv(x)
        x = self.gap(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
