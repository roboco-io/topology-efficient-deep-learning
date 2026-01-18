"""Persistence Diagram 벡터화."""

from typing import Dict, List, Optional

import numpy as np


def persistence_landscape(
    diagram: np.ndarray,
    num_landscapes: int = 5,
    resolution: int = 100,
    x_range: Optional[tuple] = None,
) -> np.ndarray:
    """
    Persistence Landscape 계산.

    Args:
        diagram: Persistence diagram (n_points, 2)
        num_landscapes: 사용할 landscape 개수
        resolution: 해상도
        x_range: (min, max) 범위

    Returns:
        Landscape 벡터 (num_landscapes * resolution,)
    """
    if len(diagram) == 0:
        return np.zeros(num_landscapes * resolution)

    births = diagram[:, 0]
    deaths = diagram[:, 1]

    if x_range is None:
        x_min = births.min()
        x_max = deaths.max()
    else:
        x_min, x_max = x_range

    x = np.linspace(x_min, x_max, resolution)
    landscapes = np.zeros((num_landscapes, resolution))

    for i, xi in enumerate(x):
        values = []
        for b, d in zip(births, deaths):
            if b <= xi <= d:
                val = min(xi - b, d - xi)
                values.append(val)
            else:
                values.append(0)

        values = sorted(values, reverse=True)
        for k in range(min(num_landscapes, len(values))):
            landscapes[k, i] = values[k]

    return landscapes.flatten()


def persistence_image(
    diagram: np.ndarray,
    resolution: int = 20,
    sigma: float = 0.1,
    x_range: Optional[tuple] = None,
    y_range: Optional[tuple] = None,
) -> np.ndarray:
    """
    Persistence Image 계산.

    Args:
        diagram: Persistence diagram (n_points, 2)
        resolution: 이미지 해상도
        sigma: 가우시안 커널 표준편차
        x_range: birth 범위
        y_range: persistence 범위

    Returns:
        Persistence image (resolution * resolution,)
    """
    if len(diagram) == 0:
        return np.zeros(resolution * resolution)

    births = diagram[:, 0]
    persistence = diagram[:, 1] - diagram[:, 0]

    if x_range is None:
        x_min, x_max = births.min(), births.max()
    else:
        x_min, x_max = x_range

    if y_range is None:
        y_min, y_max = 0, persistence.max()
    else:
        y_min, y_max = y_range

    # 패딩 추가
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    x_min -= x_pad
    x_max += x_pad
    y_max += y_pad

    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)

    image = np.zeros((resolution, resolution))

    for b, p in zip(births, persistence):
        # 가중치: persistence가 클수록 중요
        weight = p

        for i, xi in enumerate(x_grid):
            for j, yj in enumerate(y_grid):
                dist_sq = (xi - b) ** 2 + (yj - p) ** 2
                image[j, i] += weight * np.exp(-dist_sq / (2 * sigma**2))

    return image.flatten()


def persistence_statistics(diagram: np.ndarray) -> np.ndarray:
    """
    Persistence Diagram 통계량.

    Args:
        diagram: Persistence diagram (n_points, 2)

    Returns:
        통계량 벡터
    """
    if len(diagram) == 0:
        return np.zeros(10)

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    persistence = deaths - births
    midpoints = (births + deaths) / 2

    stats = [
        len(diagram),  # 포인트 개수
        births.mean(),  # birth 평균
        births.std(),  # birth 표준편차
        deaths.mean(),  # death 평균
        deaths.std(),  # death 표준편차
        persistence.mean(),  # persistence 평균
        persistence.std(),  # persistence 표준편차
        persistence.max(),  # 최대 persistence
        persistence.sum(),  # 총 persistence
        (persistence**2).sum(),  # persistence entropy 근사
    ]

    return np.array(stats)


def vectorize_diagrams(
    diagrams: Dict[int, np.ndarray],
    method: str = "persistence_landscape",
    **kwargs,
) -> np.ndarray:
    """
    여러 차원의 Persistence Diagram을 하나의 벡터로 변환.

    Args:
        diagrams: {dim: diagram} 딕셔너리
        method: 벡터화 방법
        **kwargs: 벡터화 함수 파라미터

    Returns:
        결합된 피처 벡터
    """
    vectorizers = {
        "persistence_landscape": persistence_landscape,
        "persistence_image": persistence_image,
        "statistics": persistence_statistics,
    }

    if method not in vectorizers:
        raise ValueError(f"Unknown method: {method}")

    vectorizer = vectorizers[method]
    vectors = []

    for dim in sorted(diagrams.keys()):
        vec = vectorizer(diagrams[dim], **kwargs)
        vectors.append(vec)

    return np.concatenate(vectors)
