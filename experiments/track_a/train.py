"""Track A: PH 기반 시계열 분류 실험."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.ucr import UCRDataset
from src.tda import (
    takens_embedding,
    compute_persistence_diagram,
    vectorize_diagrams,
)
from src.models.baselines import CNN1D, GRU, TCN
from src.models.tda import PHMLP
from src.utils.metrics import compute_metrics, compute_efficiency_metrics


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
        diagrams = compute_persistence_diagram(
            point_cloud, homology_dims=homology_dims
        )

        # Vectorization
        vec = vectorize_diagrams(diagrams, method=vectorization)
        features.append(vec)

    return np.array(features)


def train_epoch(model, loader, criterion, optimizer, device):
    """한 에폭 학습."""
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 데이터 로드
    train_dataset = UCRDataset(args.data_path, args.dataset, split="train")
    test_dataset = UCRDataset(args.data_path, args.dataset, split="test")

    print(f"Dataset: {args.dataset}")
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"Classes: {train_dataset.num_classes}, Seq length: {train_dataset.seq_length}")

    if args.model == "ph_mlp":
        # PH 피처 추출
        print("Extracting PH features...")
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

        # 텐서 변환
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

        model = PHMLP(
            input_dim=train_ph.shape[1],
            num_classes=train_dataset.num_classes,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        )

    else:
        # 베이스라인 모델
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        if args.model == "cnn":
            model = CNN1D(
                input_size=train_dataset.seq_length,
                num_classes=train_dataset.num_classes,
                hidden_dims=args.hidden_dims,
                dropout=args.dropout,
            )
        elif args.model == "gru":
            model = GRU(
                input_size=train_dataset.seq_length,
                num_classes=train_dataset.num_classes,
                hidden_dim=args.hidden_dims[0],
                dropout=args.dropout,
            )
        elif args.model == "tcn":
            model = TCN(
                input_size=train_dataset.seq_length,
                num_classes=train_dataset.num_classes,
                hidden_dims=args.hidden_dims,
                dropout=args.dropout,
            )

    model = model.to(device)
    print(f"Model: {args.model}, Parameters: {model.count_parameters():,}")

    # 학습
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        # 평가
        y_true, y_pred, y_prob = evaluate(model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred, y_prob)

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f} - "
                  f"F1: {metrics['f1_macro']:.4f} - Acc: {metrics['accuracy']:.4f}")

    # 최종 평가
    print("\n=== Final Results ===")
    y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    if metrics.get("auroc"):
        print(f"AUROC: {metrics['auroc']:.4f}")

    # 효율성 지표
    if args.model == "ph_mlp":
        input_shape = (1, train_ph.shape[1])
    else:
        input_shape = (1, train_dataset.seq_length)

    eff_metrics = compute_efficiency_metrics(model, input_shape, str(device))
    print(f"Parameters: {eff_metrics['params']:,}")
    print(f"Latency: {eff_metrics['latency_mean_ms']:.3f} ± {eff_metrics['latency_std_ms']:.3f} ms")
    print(f"Throughput: {eff_metrics['throughput_samples_per_sec']:.1f} samples/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_path", type=str, default="./data/ucr")
    parser.add_argument("--dataset", type=str, default="ECG200")

    # Model
    parser.add_argument("--model", type=str, default="ph_mlp",
                        choices=["ph_mlp", "cnn", "gru", "tcn"])
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--dropout", type=float, default=0.1)

    # PH
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--homology_dims", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--vectorization", type=str, default="persistence_landscape")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    args = parser.parse_args()
    main(args)
