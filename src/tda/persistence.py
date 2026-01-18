"""Persistent Homology 계산."""

from typing import List, Optional, Tuple

import numpy as np


def compute_persistence_diagram(
    point_cloud: np.ndarray,
    homology_dims: List[int] = [0, 1],
    max_edge_length: Optional[float] = None,
    backend: str = "ripser",
) -> dict:
    """
    포인트 클라우드에서 Persistence Diagram 계산.

    Args:
        point_cloud: 포인트 클라우드 (n_points, n_dims)
        homology_dims: 계산할 호몰로지 차원
        max_edge_length: 최대 엣지 길이 (None이면 자동)
        backend: "ripser" 또는 "gudhi"

    Returns:
        {dim: np.ndarray of (birth, death) pairs}
    """
    if backend == "ripser":
        return _compute_with_ripser(point_cloud, homology_dims, max_edge_length)
    elif backend == "gudhi":
        return _compute_with_gudhi(point_cloud, homology_dims, max_edge_length)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _compute_with_ripser(
    point_cloud: np.ndarray,
    homology_dims: List[int],
    max_edge_length: Optional[float],
) -> dict:
    """Ripser를 사용한 PH 계산."""
    try:
        from ripser import ripser
    except ImportError:
        raise ImportError("ripser required: pip install ripser")

    max_dim = max(homology_dims)
    thresh = max_edge_length if max_edge_length else np.inf

    result = ripser(point_cloud, maxdim=max_dim, thresh=thresh)

    diagrams = {}
    for dim in homology_dims:
        if dim < len(result["dgms"]):
            dgm = result["dgms"][dim]
            # 무한대 값 처리
            dgm = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm
            diagrams[dim] = dgm

    return diagrams


def _compute_with_gudhi(
    point_cloud: np.ndarray,
    homology_dims: List[int],
    max_edge_length: Optional[float],
) -> dict:
    """GUDHI를 사용한 PH 계산."""
    try:
        import gudhi
    except ImportError:
        raise ImportError("gudhi required: pip install gudhi")

    rips = gudhi.RipsComplex(points=point_cloud, max_edge_length=max_edge_length or 1.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=max(homology_dims) + 1)
    simplex_tree.compute_persistence()

    diagrams = {}
    for dim in homology_dims:
        pairs = simplex_tree.persistence_intervals_in_dimension(dim)
        # 무한대 값 제거
        pairs = pairs[np.isfinite(pairs[:, 1])] if len(pairs) > 0 else pairs
        diagrams[dim] = pairs

    return diagrams


def compute_cubical_persistence(
    image: np.ndarray,
    homology_dims: List[int] = [0, 1],
) -> dict:
    """
    이미지(스펙트로그램)에서 Cubical Persistence 계산.

    Args:
        image: 2D 이미지 (height, width)
        homology_dims: 계산할 호몰로지 차원

    Returns:
        {dim: np.ndarray of (birth, death) pairs}
    """
    try:
        import gudhi
    except ImportError:
        raise ImportError("gudhi required: pip install gudhi")

    cubical = gudhi.CubicalComplex(top_dimensional_cells=image)
    cubical.compute_persistence()

    diagrams = {}
    for dim in homology_dims:
        pairs = cubical.persistence_intervals_in_dimension(dim)
        pairs = pairs[np.isfinite(pairs[:, 1])] if len(pairs) > 0 else pairs
        diagrams[dim] = pairs

    return diagrams
