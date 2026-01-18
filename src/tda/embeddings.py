"""시계열을 포인트 클라우드로 변환하는 임베딩 함수."""

import numpy as np


def takens_embedding(
    time_series: np.ndarray,
    delay: int = 1,
    dimension: int = 3,
) -> np.ndarray:
    """
    Takens 지연 임베딩.

    Args:
        time_series: 1D 시계열 (n_samples,)
        delay: 지연 tau
        dimension: 임베딩 차원 d

    Returns:
        포인트 클라우드 (n_points, dimension)
    """
    n = len(time_series)
    n_points = n - (dimension - 1) * delay

    if n_points <= 0:
        raise ValueError(
            f"Time series too short for delay={delay}, dimension={dimension}"
        )

    embedded = np.zeros((n_points, dimension))
    for i in range(dimension):
        embedded[:, i] = time_series[i * delay : i * delay + n_points]

    return embedded


def sliding_window_embedding(
    time_series: np.ndarray,
    window_size: int = 32,
    stride: int = 1,
) -> np.ndarray:
    """
    슬라이딩 윈도우 임베딩.

    Args:
        time_series: 1D 시계열 (n_samples,)
        window_size: 윈도우 크기
        stride: 스트라이드

    Returns:
        포인트 클라우드 (n_windows, window_size)
    """
    n = len(time_series)
    n_windows = (n - window_size) // stride + 1

    if n_windows <= 0:
        raise ValueError(f"Time series too short for window_size={window_size}")

    embedded = np.zeros((n_windows, window_size))
    for i in range(n_windows):
        start = i * stride
        embedded[i] = time_series[start : start + window_size]

    return embedded


def stft_embedding(
    time_series: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 128,
) -> np.ndarray:
    """
    STFT 기반 스펙트로그램 임베딩.

    Args:
        time_series: 1D 시계열
        n_fft: FFT 크기
        hop_length: Hop 길이

    Returns:
        스펙트로그램 (n_freq, n_time)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required for STFT embedding: pip install librosa")

    stft = librosa.stft(time_series, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)

    return spectrogram
