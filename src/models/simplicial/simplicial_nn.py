"""Simplicial Neural Network."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplicialConv(nn.Module):
    """Simplicial Convolution Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_dim: int = 2,
    ):
        """
        Args:
            in_channels: 입력 채널 수
            out_channels: 출력 채널 수
            max_dim: 최대 simplex 차원
        """
        super().__init__()

        self.max_dim = max_dim

        # 각 차원별 변환
        self.linear = nn.ModuleDict()
        for dim in range(max_dim + 1):
            self.linear[str(dim)] = nn.Linear(in_channels, out_channels)

        # 경계/코경계 메시지 변환
        self.boundary_linear = nn.ModuleDict()
        self.coboundary_linear = nn.ModuleDict()

        for dim in range(1, max_dim + 1):
            self.boundary_linear[str(dim)] = nn.Linear(in_channels, out_channels)

        for dim in range(max_dim):
            self.coboundary_linear[str(dim)] = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x: Dict[int, torch.Tensor],
        boundaries: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Args:
            x: {dim: features} 각 차원별 simplex 피처
            boundaries: {dim: boundary_matrix} 경계 행렬

        Returns:
            업데이트된 피처
        """
        out = {}

        for dim in range(self.max_dim + 1):
            if dim not in x:
                continue

            # Self transform
            h = self.linear[str(dim)](x[dim])

            # Boundary message (from higher dim)
            if dim + 1 in boundaries and dim + 1 in x:
                B = boundaries[dim + 1]  # (dim_simplices, dim+1_simplices)
                boundary_msg = torch.mm(B, x[dim + 1])
                h = h + self.boundary_linear[str(dim + 1)](boundary_msg)

            # Coboundary message (from lower dim)
            if dim in boundaries and dim - 1 in x:
                B = boundaries[dim]  # (dim-1_simplices, dim_simplices)
                coboundary_msg = torch.mm(B.t(), x[dim - 1])
                h = h + self.coboundary_linear[str(dim - 1)](coboundary_msg)

            out[dim] = F.relu(h)

        return out


class SimplicialNN(nn.Module):
    """Simplicial Neural Network for node/graph classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        max_dim: int = 2,
        dropout: float = 0.1,
        pool: str = "mean",
    ):
        """
        Args:
            input_dim: 노드 피처 차원
            hidden_dim: 히든 차원
            output_dim: 출력 차원 (클래스 수)
            num_layers: 레이어 수
            max_dim: 최대 simplex 차원
            dropout: 드롭아웃 비율
            pool: 풀링 방식 ("mean", "sum", "max")
        """
        super().__init__()

        self.max_dim = max_dim
        self.pool = pool

        # 입력 변환
        self.input_linear = nn.Linear(input_dim, hidden_dim)

        # Simplicial convolution layers
        self.convs = nn.ModuleList([
            SimplicialConv(hidden_dim, hidden_dim, max_dim)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # 출력
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        boundaries: Dict[int, torch.Tensor],
        simplex_indices: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 노드 피처 (num_nodes, input_dim)
            boundaries: 경계 행렬
            simplex_indices: (optional) 배치 내 simplex 인덱스

        Returns:
            노드별 또는 그래프별 예측
        """
        # 노드 피처 변환
        h = {0: self.input_linear(x)}

        # Higher-dim simplex 피처 초기화 (노드 피처 평균)
        for dim in range(1, self.max_dim + 1):
            if dim in boundaries:
                B = boundaries[dim]
                # 해당 차원 simplex 수
                num_simplices = B.shape[1]
                h[dim] = torch.zeros(num_simplices, h[0].shape[1], device=x.device)

        # Simplicial convolutions
        for conv in self.convs:
            h = conv(h, boundaries)
            h = {dim: self.dropout(feat) for dim, feat in h.items()}

        # 노드 피처로 출력
        out = self.output_linear(h[0])

        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
