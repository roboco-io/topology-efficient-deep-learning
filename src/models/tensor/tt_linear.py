"""Tensor Train (TT) Linear Layer."""

from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np


def factorize_dims(n: int, num_factors: int = 4) -> List[int]:
    """정수를 num_factors개의 인수로 분해."""
    factors = []
    temp = n

    for _ in range(num_factors - 1):
        # 가장 가까운 인수 찾기
        target = int(np.power(temp, 1.0 / (num_factors - len(factors))))
        for f in range(target, 0, -1):
            if temp % f == 0:
                factors.append(f)
                temp = temp // f
                break

    factors.append(temp)

    # 크기순 정렬
    factors.sort(reverse=True)

    return factors


class TTLinear(nn.Module):
    """Tensor Train 분해 기반 Linear Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ranks: List[int] = None,
        num_cores: int = 4,
        bias: bool = True,
    ):
        """
        Args:
            in_features: 입력 차원
            out_features: 출력 차원
            ranks: TT ranks (None이면 자동 설정)
            num_cores: TT core 수
            bias: bias 사용 여부
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_cores = num_cores

        # 입출력 차원 분해
        self.in_dims = factorize_dims(in_features, num_cores)
        self.out_dims = factorize_dims(out_features, num_cores)

        # 분해된 차원으로 in/out 맞추기
        actual_in = np.prod(self.in_dims)
        actual_out = np.prod(self.out_dims)

        if actual_in != in_features:
            # 패딩 필요
            self.in_pad = actual_in - in_features
            self.in_dims[-1] = self.in_dims[-1]
        else:
            self.in_pad = 0

        if actual_out != out_features:
            self.out_pad = actual_out - out_features
        else:
            self.out_pad = 0

        # Ranks 설정
        if ranks is None:
            # 기본값: 중간 rank
            default_rank = min(16, min(in_features, out_features) // 4)
            ranks = [default_rank] * (num_cores - 1)

        self.ranks = [1] + list(ranks) + [1]

        # TT cores 생성
        self.cores = nn.ParameterList()
        for i in range(num_cores):
            core_shape = (
                self.ranks[i],
                self.in_dims[i],
                self.out_dims[i],
                self.ranks[i + 1],
            )
            core = nn.Parameter(torch.randn(*core_shape) * 0.01)
            self.cores.append(core)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features)

        Returns:
            (batch, out_features)
        """
        batch_size = x.shape[0]

        # 입력 패딩
        if self.in_pad > 0:
            x = torch.nn.functional.pad(x, (0, self.in_pad))

        # 입력을 TT 형태로 reshape
        x = x.view(batch_size, *self.in_dims)

        # TT contraction
        result = x

        for i, core in enumerate(self.cores):
            # core: (r_i, n_i, m_i, r_{i+1})
            # result: (batch, n_1, ..., n_i, r_i) after i-1 contractions

            if i == 0:
                # 첫 번째 core
                # result: (batch, n_1, n_2, ..., n_k)
                # core: (1, n_1, m_1, r_2)
                result = torch.einsum(
                    "b...i,rimo->b...mo",
                    result.unsqueeze(-1),
                    core,
                )
            else:
                # result: (batch, ..., m_{i-1}, r_i)
                # core: (r_i, n_i, m_i, r_{i+1})
                # contract along n_i and r_i

                # Reshape for contraction
                shape = result.shape
                # (..., r_i) * (r_i, n_i, m_i, r_{i+1})
                result = torch.tensordot(result, core, dims=([[-1], [0]]))

        # Reshape 결과
        result = result.view(batch_size, -1)

        # 출력 패딩 제거
        if self.out_pad > 0:
            result = result[:, :self.out_features]

        if self.bias is not None:
            result = result + self.bias

        return result

    def count_parameters(self) -> int:
        """실제 파라미터 수."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compression_ratio(self) -> float:
        """Full linear 대비 압축률."""
        full_params = self.in_features * self.out_features
        if self.bias is not None:
            full_params += self.out_features

        return full_params / self.count_parameters()
