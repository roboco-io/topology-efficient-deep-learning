#!/usr/bin/env python
"""UCR Time Series Classification Archive 다운로드 스크립트.

sktime 패키지를 사용하여 UCR 데이터셋을 다운로드합니다.
"""

import argparse
from pathlib import Path

import numpy as np

# Track A 데이터셋 목록
TRACK_A_DATASETS = [
    "ECG200",
    "FordA",
    "ElectricDevices",
    "Wafer",
    "UWaveGestureLibraryAll",
]


def download_with_sktime(dataset: str, output_dir: Path) -> bool:
    """sktime 패키지를 사용하여 데이터셋 다운로드."""
    try:
        from sktime.datasets import load_UCR_UEA_dataset
    except ImportError:
        print("sktime 패키지가 필요합니다: pip install sktime")
        return False

    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Train 데이터 다운로드
        X_train, y_train = load_UCR_UEA_dataset(dataset, split="train", return_type="numpy3D")
        # Test 데이터 다운로드
        X_test, y_test = load_UCR_UEA_dataset(dataset, split="test", return_type="numpy3D")

        # TSV 형식으로 저장 (label + values)
        for X, y, split in [(X_train, y_train, "TRAIN"), (X_test, y_test, "TEST")]:
            tsv_path = dataset_dir / f"{dataset}_{split}.tsv"

            # X shape: (n_samples, n_channels, seq_len) -> univariate일 경우 (n_samples, 1, seq_len)
            if X.ndim == 3:
                X = X.squeeze(1)  # (n_samples, seq_len)

            # NaN 처리
            X = np.nan_to_num(X, nan=0.0)

            # TSV 저장: label\tvalue1\tvalue2\t...
            with open(tsv_path, "w") as f:
                for i in range(len(X)):
                    label = y[i]
                    values = "\t".join(str(v) for v in X[i])
                    f.write(f"{label}\t{values}\n")

        print(f"  {dataset}: 다운로드 완료 (train={len(X_train)}, test={len(X_test)})")
        return True

    except Exception as e:
        print(f"  {dataset}: 다운로드 실패 - {e}")
        return False


def verify_dataset(dataset: str, output_dir: Path) -> dict:
    """다운로드된 데이터셋 검증."""
    dataset_dir = output_dir / dataset

    train_path = dataset_dir / f"{dataset}_TRAIN.tsv"
    test_path = dataset_dir / f"{dataset}_TEST.tsv"

    if not train_path.exists() or not test_path.exists():
        return {"status": "missing", "train": 0, "test": 0, "seq_len": 0}

    # 데이터 로드 테스트
    train_data = np.loadtxt(train_path, delimiter="\t", dtype=str)
    test_data = np.loadtxt(test_path, delimiter="\t", dtype=str)

    # 클래스 추출 (첫 번째 열)
    train_labels = train_data[:, 0]

    return {
        "status": "ok",
        "train": len(train_data),
        "test": len(test_data),
        "seq_len": train_data.shape[1] - 1,  # label 제외
        "classes": len(np.unique(train_labels)),
    }


def main():
    parser = argparse.ArgumentParser(description="UCR 데이터셋 다운로드")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=TRACK_A_DATASETS,
        help="다운로드할 데이터셋 목록 (기본: Track A 데이터셋)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/ucr",
        help="출력 디렉토리 (기본: ./data/ucr)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UCR Time Series Classification Archive 다운로드")
    print("=" * 60)
    print(f"대상 데이터셋: {args.datasets}")
    print(f"출력 디렉토리: {output_dir.absolute()}")
    print()

    # 다운로드
    print("데이터셋 다운로드 중...")
    success = []
    failed = []

    for dataset in args.datasets:
        print(f"\n{dataset}:")
        if download_with_sktime(dataset, output_dir):
            success.append(dataset)
        else:
            failed.append(dataset)

    # 검증
    print("\n" + "=" * 60)
    print("다운로드 검증")
    print("=" * 60)

    for dataset in args.datasets:
        info = verify_dataset(dataset, output_dir)
        if info["status"] == "ok":
            print(
                f"  {dataset}: train={info['train']}, test={info['test']}, "
                f"seq_len={info['seq_len']}, classes={info['classes']}"
            )
        else:
            print(f"  {dataset}: 누락")

    # 요약
    print("\n" + "=" * 60)
    print("다운로드 완료")
    print("=" * 60)
    print(f"성공: {len(success)}/{len(args.datasets)}")
    if failed:
        print(f"실패: {failed}")


if __name__ == "__main__":
    main()
