"""SageMaker용 Track A 학습 스크립트.

SageMaker Training Job에서 실행되는 엔트리포인트.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# SageMaker 환경 변수
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============== TDA Functions ==============

def takens_embedding(time_series: np.ndarray, delay: int = 1, dimension: int = 3, max_points: int = 300) -> np.ndarray:
    """Takens embedding with optional subsampling for long sequences."""
    n = len(time_series)
    n_points = n - (dimension - 1) * delay
    if n_points <= 0:
        raise ValueError(f"Time series too short for delay={delay}, dimension={dimension}")
    embedded = np.zeros((n_points, dimension))
    for i in range(dimension):
        embedded[:, i] = time_series[i * delay : i * delay + n_points]

    # Subsample if too many points (for PH computation efficiency)
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
        values = [max(0, min(xi - b, d - xi)) for b, d in zip(births, deaths)]
        values = sorted(values, reverse=True)
        for k in range(min(num_landscapes, len(values))):
            landscapes[k, i] = values[k]
    return landscapes.flatten()


def vectorize_diagrams(diagrams: dict, method: str = "persistence_landscape") -> np.ndarray:
    vectors = []
    for dim in sorted(diagrams.keys()):
        vec = persistence_landscape(diagrams[dim])
        vectors.append(vec)
    return np.concatenate(vectors)


def _extract_single_ph(args):
    """Single sample PH extraction for parallel processing."""
    sample, delay, dimension, homology_dims = args
    pc = takens_embedding(sample, delay=delay, dimension=dimension)
    dg = compute_persistence_diagram(pc, homology_dims=homology_dims)
    return vectorize_diagrams(dg)


def extract_ph_features(data: np.ndarray, delay: int, dimension: int, homology_dims: list, n_jobs: int = -1) -> np.ndarray:
    """Extract PH features with parallel processing."""
    from joblib import Parallel, delayed

    # Prepare arguments for each sample
    args_list = [(sample, delay, dimension, homology_dims) for sample in data]

    # Parallel extraction (-1 uses all available CPUs)
    features = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_extract_single_ph)(args) for args in args_list
    )

    return np.array(features)


# ============== Models ==============

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


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, n_filters: int = 32, kernel_sizes: list = [9, 19, 39], bottleneck_channels: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False) if in_channels > 1 else None
        conv_in = bottleneck_channels if self.bottleneck else in_channels
        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, n_filters, k, padding=k // 2, bias=False) for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_maxpool = nn.Conv1d(in_channels, n_filters, 1, bias=False)
        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))

    def forward(self, x):
        x_bn = self.bottleneck(x) if self.bottleneck else x
        conv_outs = [conv(x_bn) for conv in self.convs]
        mp_out = self.conv_maxpool(self.maxpool(x))
        out = torch.cat(conv_outs + [mp_out], dim=1)
        return torch.relu(self.bn(out))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, n_filters: int = 32, kernel_sizes: list = [9, 19, 39]):
        super().__init__()
        out_channels = n_filters * (len(kernel_sizes) + 1)
        self.inception1 = InceptionModule(in_channels, n_filters, kernel_sizes)
        self.inception2 = InceptionModule(out_channels, n_filters, kernel_sizes)
        self.inception3 = InceptionModule(out_channels, n_filters, kernel_sizes)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        out = self.inception3(self.inception2(self.inception1(x)))
        return torch.relu(out + self.shortcut(x))


class InceptionTime(nn.Module):
    def __init__(self, input_size: int, num_classes: int, n_filters: int = 32, n_blocks: int = 6, dropout: float = 0.0):
        super().__init__()
        out_channels = n_filters * 4
        blocks = [InceptionBlock(1 if i == 0 else out_channels, n_filters) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.dropout(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============== Training ==============

def load_data(data_dir: str, dataset: str):
    """TSV 형식 데이터 로드."""
    train_path = Path(data_dir) / dataset / f"{dataset}_TRAIN.tsv"
    test_path = Path(data_dir) / dataset / f"{dataset}_TEST.tsv"

    train_data = np.loadtxt(train_path, delimiter="\t", dtype=str)
    test_data = np.loadtxt(test_path, delimiter="\t", dtype=str)

    train_labels = train_data[:, 0]
    train_X = train_data[:, 1:].astype(np.float32)
    test_labels = test_data[:, 0]
    test_X = test_data[:, 1:].astype(np.float32)

    # Label encoding
    unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    train_y = np.array([label_map[l] for l in train_labels])
    test_y = np.array([label_map[l] for l in test_labels])

    return train_X, train_y, test_X, test_y, len(unique_labels)


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


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}, Dataset: {args.dataset}, Seed: {args.seed}")

    # Load data
    train_X, train_y, test_X, test_y, num_classes = load_data(SM_CHANNEL_TRAINING, args.dataset)
    print(f"Train: {len(train_X)}, Test: {len(test_X)}, Classes: {num_classes}, SeqLen: {train_X.shape[1]}")

    # Prepare data
    if args.model == "ph_mlp":
        print("Extracting PH features...")
        start = time.time()
        train_ph = extract_ph_features(train_X, args.delay, args.dimension, args.homology_dims)
        test_ph = extract_ph_features(test_X, args.delay, args.dimension, args.homology_dims)
        print(f"PH extraction: {time.time() - start:.1f}s, dim={train_ph.shape[1]}")

        train_dataset = TensorDataset(torch.tensor(train_ph, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(test_ph, dtype=torch.float32), torch.tensor(test_y, dtype=torch.long))
        input_dim = train_ph.shape[1]
    else:
        train_dataset = TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(test_X, dtype=torch.float32), torch.tensor(test_y, dtype=torch.long))
        input_dim = train_X.shape[1]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    if args.model == "ph_mlp":
        model = PHMLP(input_dim, num_classes, [64, 32], args.dropout)
    else:
        model = InceptionTime(input_dim, num_classes, dropout=args.dropout)

    model = model.to(device)
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
            best_f1 = metrics["f1_macro"]
            best_epoch = epoch + 1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}, f1={metrics['f1_macro']:.4f}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Final evaluation
    if best_state:
        model.load_state_dict(best_state)

    y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    final_metrics = compute_metrics(y_true, y_pred, y_prob)

    # Measure latency
    model.eval()
    dummy = torch.randn(1, input_dim).to(device)
    if args.model != "ph_mlp":
        dummy = dummy.unsqueeze(1)

    for _ in range(10):
        model(dummy)

    latencies = []
    with torch.no_grad():
        for _ in range(100):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    # Results
    results = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "metrics": {
            "accuracy": float(final_metrics["accuracy"]),
            "f1_macro": float(final_metrics["f1_macro"]),
            "auroc": float(final_metrics.get("auroc", 0)),
        },
        "efficiency": {
            "params": model.count_parameters(),
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_std_ms": float(np.std(latencies)),
        },
    }

    print("\n" + "=" * 50)
    print(f"Final Results - {args.dataset} / {args.model}")
    print("=" * 50)
    print(f"Best Epoch: {best_epoch}")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"F1 (macro): {final_metrics['f1_macro']:.4f}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Latency: {np.mean(latencies):.3f} ms")

    # Save results
    os.makedirs(SM_OUTPUT_DATA_DIR, exist_ok=True)
    with open(os.path.join(SM_OUTPUT_DATA_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    torch.save(model.state_dict(), os.path.join(SM_MODEL_DIR, "model.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ECG200")
    parser.add_argument("--model", type=str, default="ph_mlp", choices=["ph_mlp", "inceptiontime"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--homology_dims", type=int, nargs="+", default=[0, 1])

    args = parser.parse_args()
    main(args)
