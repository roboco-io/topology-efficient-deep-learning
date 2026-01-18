"""평가 지표 계산."""

import time
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    task: str = "classification",
) -> Dict[str, float]:
    """
    성능 지표 계산.

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        y_prob: 예측 확률 (AUROC용)
        task: "classification" 또는 "regression"

    Returns:
        지표 딕셔너리
    """
    if task == "classification":
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        }

        if y_prob is not None:
            try:
                if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                    # Binary classification
                    prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                    metrics["auroc"] = roc_auc_score(y_true, prob)
                else:
                    # Multi-class
                    metrics["auroc"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="macro"
                    )
            except ValueError:
                metrics["auroc"] = None

    elif task == "regression":
        metrics = {
            "mae": np.mean(np.abs(y_true - y_pred)),
            "mse": np.mean((y_true - y_pred) ** 2),
            "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        }

    return metrics


def compute_efficiency_metrics(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    효율성 지표 계산.

    Args:
        model: PyTorch 모델
        input_shape: 입력 shape (batch_size 포함)
        device: 디바이스
        num_runs: 측정 실행 횟수
        warmup_runs: 워밍업 실행 횟수

    Returns:
        효율성 지표 딕셔너리
    """
    model = model.to(device)
    model.eval()

    # 파라미터 수
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 더미 입력 생성
    dummy_input = torch.randn(*input_shape, device=device)

    # 워밍업
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # GPU 동기화
    if device == "cuda":
        torch.cuda.synchronize()

    # Latency 측정
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    # VRAM 측정
    peak_vram = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    metrics = {
        "params": params,
        "latency_mean_ms": np.mean(latencies),
        "latency_std_ms": np.std(latencies),
        "throughput_samples_per_sec": input_shape[0] / (np.mean(latencies) / 1000),
    }

    if peak_vram is not None:
        metrics["peak_vram_mb"] = peak_vram

    return metrics
