"""Track A: PH 기반 시계열 분류 실험.

Usage:
    python experiments/track_a/train.py --model ph_mlp --dataset ECG200
    python experiments/track_a/train.py --model inceptiontime --dataset ECG200 --use_wandb
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.ucr import UCRDataset
from src.models.baselines import CNN1D, FCN, GRU, InceptionTime, ResNet1D, TCN
from src.models.tda import PHMLP
from src.tda import (
    compute_persistence_diagram,
    takens_embedding,
    vectorize_diagrams,
)
from src.utils.metrics import compute_efficiency_metrics, compute_metrics

# wandb 선택적 임포트
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def set_seed(seed: int):
    """재현성을 위한 시드 설정."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_ph_features(
    data: np.ndarray,
    delay: int = 5,
    dimension: int = 3,
    homology_dims: list = [0, 1],
    vectorization: str = "persistence_landscape",
) -> np.ndarray:
    """시계열 데이터에서 PH 피처 추출."""
    features = []

    for sample in data:
        # Takens embedding
        point_cloud = takens_embedding(sample, delay=delay, dimension=dimension)

        # Persistence diagram
        diagrams = compute_persistence_diagram(point_cloud, homology_dims=homology_dims)

        # Vectorization
        vec = vectorize_diagrams(diagrams, method=vectorization)
        features.append(vec)

    return np.array(features)


def measure_ph_latency(
    data: np.ndarray,
    delay: int,
    dimension: int,
    homology_dims: list,
    vectorization: str,
    n_runs: int = 100,
) -> dict:
    """PH 피처 추출 레이턴시 측정."""
    sample = data[0]

    # Warmup
    for _ in range(10):
        pc = takens_embedding(sample, delay=delay, dimension=dimension)
        dg = compute_persistence_diagram(pc, homology_dims=homology_dims)
        vectorize_diagrams(dg, method=vectorization)

    # Measure
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        pc = takens_embedding(sample, delay=delay, dimension=dimension)
        dg = compute_persistence_diagram(pc, homology_dims=homology_dims)
        vectorize_diagrams(dg, method=vectorization)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "ph_latency_mean_ms": np.mean(latencies),
        "ph_latency_std_ms": np.std(latencies),
    }


def create_model(args, input_dim: int, num_classes: int) -> nn.Module:
    """모델 생성."""
    if args.model == "ph_mlp":
        return PHMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        )
    elif args.model == "inceptiontime":
        return InceptionTime(
            input_size=input_dim,
            num_classes=num_classes,
            n_filters=32,
            n_blocks=6,
            dropout=args.dropout,
        )
    elif args.model == "resnet":
        return ResNet1D(
            input_size=input_dim,
            num_classes=num_classes,
            hidden_dims=[64, 128, 128],
            dropout=args.dropout,
        )
    elif args.model == "fcn":
        return FCN(
            input_size=input_dim,
            num_classes=num_classes,
            hidden_dims=[128, 256, 128],
            dropout=args.dropout,
        )
    elif args.model == "cnn":
        return CNN1D(
            input_size=input_dim,
            num_classes=num_classes,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        )
    elif args.model == "gru":
        return GRU(
            input_size=input_dim,
            num_classes=num_classes,
            hidden_dim=args.hidden_dims[0],
            dropout=args.dropout,
        )
    elif args.model == "tcn":
        return TCN(
            input_size=input_dim,
            num_classes=num_classes,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")


def train_epoch(model, loader, criterion, optimizer, device):
    """한 에폭 학습."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total,
    }


def evaluate(model, loader, device):
    """평가."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            output = model(x)
            probs = torch.softmax(output, dim=1)

            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def get_device():
    """사용 가능한 최적의 디바이스 반환."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main(args):
    # 시드 설정
    set_seed(args.seed)

    device = get_device()
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # 결과 디렉토리
    run_name = f"{args.dataset}_{args.model}_seed{args.seed}"
    if args.model == "ph_mlp":
        run_name += f"_{args.vectorization}"

    result_dir = Path(args.output_dir) / args.dataset / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # wandb 초기화
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            group="track_a",
            name=run_name,
            config=vars(args),
            tags=[args.dataset, args.model],
        )

    # 데이터 로드
    train_dataset = UCRDataset(args.data_path, args.dataset, split="train")
    test_dataset = UCRDataset(args.data_path, args.dataset, split="test")

    print(f"Dataset: {args.dataset}")
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"Classes: {train_dataset.num_classes}, Seq length: {train_dataset.seq_length}")

    # PH 피처 추출 (ph_mlp인 경우)
    ph_latency_metrics = {}
    if args.model == "ph_mlp":
        print("Extracting PH features...")
        start_time = time.time()

        train_ph = extract_ph_features(
            train_dataset.data,
            delay=args.delay,
            dimension=args.dimension,
            homology_dims=args.homology_dims,
            vectorization=args.vectorization,
        )
        test_ph = extract_ph_features(
            test_dataset.data,
            delay=args.delay,
            dimension=args.dimension,
            homology_dims=args.homology_dims,
            vectorization=args.vectorization,
        )

        extraction_time = time.time() - start_time
        print(f"PH feature extraction: {extraction_time:.2f}s")
        print(f"Feature dimension: {train_ph.shape[1]}")

        # PH 레이턴시 측정
        ph_latency_metrics = measure_ph_latency(
            train_dataset.data,
            args.delay,
            args.dimension,
            args.homology_dims,
            args.vectorization,
        )
        print(f"PH latency: {ph_latency_metrics['ph_latency_mean_ms']:.3f} ms/sample")

        # 데이터 준비
        train_x = torch.from_numpy(train_ph).float()
        test_x = torch.from_numpy(test_ph).float()
        train_y = torch.from_numpy(train_dataset.labels).long()
        test_y = torch.from_numpy(test_dataset.labels).long()

        train_loader = DataLoader(
            list(zip(train_x, train_y)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            list(zip(test_x, test_y)),
            batch_size=args.batch_size,
        )

        input_dim = train_ph.shape[1]
    else:
        # 베이스라인 모델용 데이터
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        input_dim = train_dataset.seq_length

    # 모델 생성
    model = create_model(args, input_dim, train_dataset.num_classes)
    model = model.to(device)
    print(f"Model: {args.model}, Parameters: {model.count_parameters():,}")

    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Early stopping
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # 학습 로그
    training_log = []

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        # Evaluate
        y_true, y_pred, y_prob = evaluate(model, test_loader, device)
        test_metrics = compute_metrics(y_true, y_pred, y_prob)

        # Logging
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "test_acc": test_metrics["accuracy"],
            "test_f1": test_metrics["f1_macro"],
            "lr": scheduler.get_last_lr()[0],
        }
        training_log.append(log_entry)

        # wandb logging
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "test/accuracy": test_metrics["accuracy"],
                    "test/f1_macro": test_metrics["f1_macro"],
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch + 1,
            )

        # Early stopping check
        if test_metrics["f1_macro"] > best_f1:
            best_f1 = test_metrics["f1_macro"]
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"Loss: {train_metrics['loss']:.4f} - "
                f"F1: {test_metrics['f1_macro']:.4f} - "
                f"Acc: {test_metrics['accuracy']:.4f}"
            )

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 최적 모델 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 최종 평가
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    final_metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"Best Epoch: {best_epoch}")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"F1 (macro): {final_metrics['f1_macro']:.4f}")
    if final_metrics.get("auroc"):
        print(f"AUROC: {final_metrics['auroc']:.4f}")

    # 효율성 지표
    if args.model == "ph_mlp":
        input_shape = (1, train_ph.shape[1])
    else:
        input_shape = (1, train_dataset.seq_length)

    eff_metrics = compute_efficiency_metrics(model, input_shape, str(device))
    print(f"\nParameters: {eff_metrics['params']:,}")
    print(
        f"Model latency: {eff_metrics['latency_mean_ms']:.3f} ± "
        f"{eff_metrics['latency_std_ms']:.3f} ms"
    )
    print(f"Throughput: {eff_metrics['throughput_samples_per_sec']:.1f} samples/s")

    # PH 포함 총 레이턴시
    if args.model == "ph_mlp":
        total_latency = (
            eff_metrics["latency_mean_ms"] + ph_latency_metrics["ph_latency_mean_ms"]
        )
        print(f"\nTotal latency (with PH): {total_latency:.3f} ms")

    # 결과 저장
    results = {
        "config": vars(args),
        "best_epoch": best_epoch,
        "metrics": {
            "accuracy": float(final_metrics["accuracy"]),
            "f1_macro": float(final_metrics["f1_macro"]),
            "f1_weighted": float(final_metrics.get("f1_weighted", 0)),
            "auroc": float(final_metrics.get("auroc", 0)),
        },
        "efficiency": {
            "params": eff_metrics["params"],
            "model_latency_mean_ms": eff_metrics["latency_mean_ms"],
            "model_latency_std_ms": eff_metrics["latency_std_ms"],
            "throughput_samples_per_sec": eff_metrics["throughput_samples_per_sec"],
        },
    }

    if args.model == "ph_mlp":
        results["efficiency"]["ph_latency_mean_ms"] = ph_latency_metrics[
            "ph_latency_mean_ms"
        ]
        results["efficiency"]["ph_latency_std_ms"] = ph_latency_metrics[
            "ph_latency_std_ms"
        ]
        results["efficiency"]["total_latency_ms"] = total_latency

    # 파일 저장
    with open(result_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(result_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    torch.save(model.state_dict(), result_dir / "model.pt")

    print(f"\nResults saved to: {result_dir}")

    # wandb 종료
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log(
            {
                "final/accuracy": final_metrics["accuracy"],
                "final/f1_macro": final_metrics["f1_macro"],
                "efficiency/params": eff_metrics["params"],
                "efficiency/latency_ms": eff_metrics["latency_mean_ms"],
                "efficiency/throughput": eff_metrics["throughput_samples_per_sec"],
            }
        )
        if args.model == "ph_mlp":
            wandb.log(
                {
                    "efficiency/ph_latency_ms": ph_latency_metrics["ph_latency_mean_ms"],
                    "efficiency/total_latency_ms": total_latency,
                }
            )
        wandb.finish()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A: PH-based Time Series Classification")

    # Data
    parser.add_argument("--data_path", type=str, default="./data/ucr")
    parser.add_argument("--dataset", type=str, default="ECG200")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="ph_mlp",
        choices=["ph_mlp", "inceptiontime", "resnet", "fcn", "cnn", "gru", "tcn"],
    )
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--dropout", type=float, default=0.1)

    # PH parameters
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--homology_dims", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--vectorization",
        type=str,
        default="persistence_landscape",
        choices=["persistence_landscape", "persistence_image", "statistics"],
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", type=str, default="./results/track_a")

    # wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="topology-efficient-dl")

    args = parser.parse_args()
    main(args)
