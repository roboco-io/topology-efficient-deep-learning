"""PH 피처 기반 MLP 모델."""

from typing import List

import torch
import torch.nn as nn


class PHMLP(nn.Module):
    """Persistent Homology 피처를 사용한 MLP 분류기."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: PH 벡터화 피처 차원
            num_classes: 분류 클래스 수
            hidden_dims: 히든 레이어 차원
            dropout: 드롭아웃 비율
        """
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: PH 벡터화 피처 (batch, input_dim)

        Returns:
            로짓 (batch, num_classes)
        """
        return self.mlp(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
