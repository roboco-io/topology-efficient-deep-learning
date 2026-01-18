"""CNN + PH 하이브리드 모델."""

from typing import List

import torch
import torch.nn as nn

from ..baselines.cnn import CNN1D


class PHHybrid(nn.Module):
    """CNN 인코더 + PH 피처 + MLP 헤드."""

    def __init__(
        self,
        seq_length: int,
        ph_dim: int,
        num_classes: int,
        encoder_dims: List[int] = [16, 32],
        head_dims: List[int] = [32],
        dropout: float = 0.1,
    ):
        """
        Args:
            seq_length: 입력 시계열 길이
            ph_dim: PH 벡터화 피처 차원
            num_classes: 분류 클래스 수
            encoder_dims: CNN 인코더 채널 수
            head_dims: MLP 헤드 히든 차원
            dropout: 드롭아웃 비율
        """
        super().__init__()

        # CNN 인코더 (분류 헤드 없이)
        self.encoder = nn.Sequential()
        in_channels = 1

        for out_channels in encoder_dims:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                )
            )
            in_channels = out_channels

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # PH 피처와 결합
        combined_dim = encoder_dims[-1] + ph_dim

        # MLP 헤드
        layers = []
        in_dim = combined_dim

        for hidden_dim in head_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.head = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        ph_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: 원본 시계열 (batch, seq_len)
            ph_features: PH 벡터화 피처 (batch, ph_dim)

        Returns:
            로짓 (batch, num_classes)
        """
        # CNN 인코딩
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.encoder(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)

        # PH 피처와 결합
        x = torch.cat([x, ph_features], dim=1)

        # 분류
        x = self.head(x)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
