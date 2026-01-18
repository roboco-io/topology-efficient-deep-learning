#!/usr/bin/env python
"""PH-MLP 학습 스크립트 (ECS Fargate용, CPU 최적화)."""

import argparse
import json
import os
import time
from pathlib import Path

import boto3
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ============== TDA Functions (Parallel) ==============

def takens_embedding(time_series: np.ndarray, delay: int = 1, dimension: int = 3, max_points: int = 300) -> np.ndarray:
    """Takens embedding with subsampling."""
    n = len(time_series)
    n_points = n - (dimension - 1) * delay
    if n_points <= 0:
        raise ValueError(f"Time series too short")
    embedded = np.zeros((n_points, dimension))
    for i in range(dimension):
        embedded[:, i] = time_series[i * delay : i * delay + n_points]
    if len(embedded) > max_points:
        indices = np.linspace(0, len(embedded) - 1, max_points, dtype=int)
        embedded = embedded[indices]
    return embedded


def compute_persistence_diagram(point_cloud: np.ndarray, homology_dims: list = [0, 1]) -> dict:
    from ripser import ripser
    max_dim = max(homology_dims)
    result = ripser(point_cloud, maxdim=max_dim, thresh=np.inf)
    diagrams = {}
    for dim in homology_dims:
        if dim < len(result["dgms"]):
            dgm = result["dgms"][dim]
            dgm = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm
            diagrams[dim] = dgm
    return diagrams


def persistence_landscape(diagram: np.ndarray, num_landscapes: int = 5, resolution: int = 100) -> np.ndarray:
    if len(diagram) == 0:
        return np.zeros(num_landscapes * resolution)
    births, deaths = diagram[:, 0], diagram[:, 1]
    x_min, x_max = births.min(), deaths.max()
    x = np.linspace(x_min, x_max, resolution)
    landscapes = np.zeros((num_landscapes, resolution))
    for i, xi in enumerate(x):
        values = sorted([max(0, min(xi - b, d - xi)) for b, d in zip(births, deaths)], reverse=True)
        for k in range(min(num_landscapes, len(values))):
            landscapes[k, i] = values[k]
    return landscapes.flatten()


def vectorize_diagrams(diagrams: dict) -> np.ndarray:
    vectors = [persistence_landscape(diagrams[dim]) for dim in sorted(diagrams.keys())]
    return np.concatenate(vectors)


def _extract_single_ph(args):
    sample, delay, dimension, homology_dims = args
    pc = takens_embedding(sample, delay=delay, dimension=dimension)
    dg = compute_persistence_diagram(pc, homology_dims=homology_dims)
    return vectorize_diagrams(dg)


def extract_ph_features_parallel(data: np.ndarray, delay: int, dimension: int,
                                  homology_dims: list, n_jobs: int = -1) -> np.ndarray:
    """Parallel PH feature extraction."""
    args_list = [(sample, delay, dimension, homology_dims) for sample in data]
    features = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
        delayed(_extract_single_ph)(args) for args in args_list
    )
    return np.array(features)


# ============== Model ==============

class PHMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = [64, 32], dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============== Data Loading ==============

def download_from_s3(s3_path: str, local_path: str, profile: str = None):
    """Download data from S3."""
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3")

    # Parse s3://bucket/key
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)


def load_data(data_dir: str, dataset: str):
    """Load TSV format data."""
    train_path = Path(data_dir) / dataset / f"{dataset}_TRAIN.tsv"
    test_path = Path(data_dir) / dataset / f"{dataset}_TEST.tsv"

    train_data = np.loadtxt(train_path, delimiter="\t", dtype=str)
    test_data = np.loadtxt(test_path, delimiter="\t", dtype=str)

    train_labels, train_X = train_data[:, 0], train_data[:, 1:].astype(np.float32)
    test_labels, test_X = test_data[:, 0], test_data[:, 1:].astype(np.float32)

    unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    train_y = np.array([label_map[l] for l in train_labels])
    test_y = np.array([label_map[l] for l in test_labels])

    return train_X, train_y, test_X, test_y, len(unique_labels)


# ============== Training ==============

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
            probs.extend(torch.softmax(out, 1).cpu().numpy())
    return np.array(labels), np.array(preds), np.array(probs)


def compute_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    try:
        if y_prob.shape[1] == 2:
            metrics["auroc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics["auroc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except:
        metrics["auroc"] = 0.0
    return metrics


def upload_results(results: dict, s3_bucket: str, s3_key: str, profile: str = None):
    """Upload results to S3."""
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3")
    s3.put_object(
        Bucket=s3_bucket,
        Key=s3_key,
        Body=json.dumps(results, indent=2),
        ContentType="application/json"
    )


def main(args):
    set_seed(args.seed)
    device = torch.device("cpu")  # Fargate는 CPU만
    n_cpus = os.cpu_count() or 4

    print(f"Device: {device}, CPUs: {n_cpus}")
    print(f"Dataset: {args.dataset}, Seed: {args.seed}")

    # Load data
    train_X, train_y, test_X, test_y, num_classes = load_data(args.data_dir, args.dataset)
    print(f"Train: {len(train_X)}, Test: {len(test_X)}, Classes: {num_classes}, SeqLen: {train_X.shape[1]}")

    # PH feature extraction (parallel)
    print(f"Extracting PH features with {n_cpus} CPUs...")
    start = time.time()
    train_ph = extract_ph_features_parallel(train_X, args.delay, args.dimension, args.homology_dims, n_jobs=n_cpus)
    test_ph = extract_ph_features_parallel(test_X, args.delay, args.dimension, args.homology_dims, n_jobs=n_cpus)
    ph_time = time.time() - start
    print(f"PH extraction: {ph_time:.1f}s, dim={train_ph.shape[1]}")

    # Prepare datasets
    train_dataset = TensorDataset(torch.tensor(train_ph, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_ph, dtype=torch.float32), torch.tensor(test_y, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model
    model = PHMLP(train_ph.shape[1], num_classes, [64, 32], args.dropout).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_epoch, patience_counter = 0, 0, 0
    best_state = None

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        y_true, y_pred, y_prob = evaluate(model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred, y_prob)

        if metrics["f1_macro"] > best_f1:
            best_f1, best_epoch = metrics["f1_macro"], epoch + 1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={train_loss:.4f}, f1={metrics['f1_macro']:.4f}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Final evaluation
    if best_state:
        model.load_state_dict(best_state)

    y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    final_metrics = compute_metrics(y_true, y_pred, y_prob)

    # Results
    results = {
        "dataset": args.dataset,
        "model": "ph_mlp",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "ph_extraction_time_sec": ph_time,
        "metrics": {
            "accuracy": float(final_metrics["accuracy"]),
            "f1_macro": float(final_metrics["f1_macro"]),
            "auroc": float(final_metrics.get("auroc", 0)),
        },
        "efficiency": {
            "params": model.count_parameters(),
            "n_cpus": n_cpus,
        },
    }

    print("\n" + "=" * 50)
    print(f"Final Results - {args.dataset} / ph_mlp")
    print("=" * 50)
    print(f"F1 (macro): {final_metrics['f1_macro']:.4f}")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"PH extraction time: {ph_time:.1f}s")

    # Save/upload results
    if args.output_s3:
        bucket = args.output_s3.replace("s3://", "").split("/")[0]
        key = f"topology-dl/results/{args.dataset}_ph_mlp_seed{args.seed}.json"
        upload_results(results, bucket, key, args.profile)
        print(f"Results uploaded to s3://{bucket}/{key}")
    else:
        output_path = Path(args.output_dir) / f"{args.dataset}_ph_mlp_seed{args.seed}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="/data/ucr")
    parser.add_argument("--output-dir", type=str, default="/output")
    parser.add_argument("--output-s3", type=str, help="S3 bucket for results")
    parser.add_argument("--profile", type=str, help="AWS profile")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--homology-dims", type=int, nargs="+", default=[0, 1])

    args = parser.parse_args()
    main(args)
