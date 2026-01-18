"""InceptionTime 베이스라인 모델.

Reference: Fawaz et al., "InceptionTime: Finding AlexNet for Time Series Classification" (2020)
https://arxiv.org/abs/1909.04939
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """Inception 모듈: 다양한 커널 크기의 병렬 합성곱."""

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes: List[int] = [9, 19, 39],
        bottleneck_channels: int = 32,
        use_bottleneck: bool = True,
    ):
        super().__init__()

        self.use_bottleneck = use_bottleneck

        if use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
            conv_in_channels = bottleneck_channels
        else:
            self.bottleneck = None
            conv_in_channels = in_channels

        # 다양한 커널 크기의 합성곱
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.convs.append(
                nn.Conv1d(conv_in_channels, n_filters, k, padding=padding, bias=False)
            )

        # Max pooling 브랜치
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_maxpool = nn.Conv1d(in_channels, n_filters, 1, bias=False)

        # 전체 출력 채널: n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottleneck
        if self.bottleneck is not None:
            x_bottleneck = self.bottleneck(x)
        else:
            x_bottleneck = x

        # 다양한 커널 크기 합성곱
        conv_outputs = [conv(x_bottleneck) for conv in self.convs]

        # Max pooling 브랜치
        maxpool_out = self.conv_maxpool(self.maxpool(x))

        # 채널 차원으로 연결
        out = torch.cat(conv_outputs + [maxpool_out], dim=1)
        out = self.bn(out)
        out = F.relu(out)

        return out


class InceptionBlock(nn.Module):
    """Inception Block: 3개의 Inception 모듈 + Residual 연결."""

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes: List[int] = [9, 19, 39],
        use_residual: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual
        n_branches = len(kernel_sizes) + 1  # +1 for maxpool branch
        out_channels = n_filters * n_branches

        # 3개의 Inception 모듈
        self.inception1 = InceptionModule(in_channels, n_filters, kernel_sizes)
        self.inception2 = InceptionModule(out_channels, n_filters, kernel_sizes)
        self.inception3 = InceptionModule(out_channels, n_filters, kernel_sizes)

        # Residual 연결
        if use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inception1(x)
        out = self.inception2(out)
        out = self.inception3(out)

        if self.use_residual:
            out = out + self.shortcut(x)
            out = F.relu(out)

        return out


class InceptionTime(nn.Module):
    """InceptionTime: 시계열 분류를 위한 앙상블 기반 모델.

    원본 논문에서는 5개의 InceptionTime 모델을 앙상블하지만,
    여기서는 단일 모델로 구현.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        n_filters: int = 32,
        n_blocks: int = 6,
        kernel_sizes: List[int] = [9, 19, 39],
        use_residual: bool = True,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_size: 시계열 길이
            num_classes: 클래스 수
            n_filters: 각 Inception 브랜치의 필터 수
            n_blocks: Inception 블록 수 (논문에서는 6개)
            kernel_sizes: Inception 모듈의 커널 크기 리스트
            use_residual: Residual 연결 사용 여부
            dropout: Dropout 비율
        """
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        n_branches = len(kernel_sizes) + 1
        out_channels = n_filters * n_branches

        # Inception 블록 스택
        blocks = []
        for i in range(n_blocks):
            in_ch = 1 if i == 0 else out_channels
            blocks.append(InceptionBlock(in_ch, n_filters, kernel_sizes, use_residual))

        self.blocks = nn.Sequential(*blocks)

        # Global Average Pooling + Classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(out_channels, num_classes)

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
