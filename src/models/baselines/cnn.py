"""1D CNN 베이스라인 모델."""

from typing import List

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for time series classification."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_dims: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_channels = 1

        for out_channels in hidden_dims:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) -> (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
