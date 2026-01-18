"""모델 프로파일링."""

from typing import Dict, Optional

import torch
import torch.nn as nn


def profile_model(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, any]:
    """
    모델 프로파일링.

    Args:
        model: PyTorch 모델
        input_shape: 입력 shape
        device: 디바이스

    Returns:
        프로파일링 결과
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    results = {
        "params": count_parameters(model),
        "params_by_layer": count_parameters_by_layer(model),
    }

    # FLOPs 계산 (fvcore 사용 가능 시)
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(model, dummy_input)
        results["flops"] = flops.total()
        results["flops_by_operator"] = dict(flops.by_operator())
    except ImportError:
        results["flops"] = None
        results["flops_by_operator"] = None

    # 메모리 분석
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        results["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

    return results


def count_parameters(model: nn.Module) -> int:
    """학습 가능한 파라미터 수."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """레이어별 파라미터 수."""
    params_by_layer = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                params_by_layer[name] = params
    return params_by_layer


def estimate_flops_linear(in_features: int, out_features: int, batch_size: int) -> int:
    """Linear 레이어 FLOPs 추정."""
    # 곱셈: batch * in * out, 덧셈: batch * (in-1) * out
    return batch_size * (2 * in_features - 1) * out_features


def estimate_flops_conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    seq_length: int,
    batch_size: int,
) -> int:
    """Conv1D FLOPs 추정."""
    return batch_size * out_channels * seq_length * (2 * in_channels * kernel_size - 1)
