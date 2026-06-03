from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold

from prediction_io import aggregate_rows, ensure_binary_probabilities, write_prediction_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild EdgeConv prediction rows from saved fold logits without touching the original source tree."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with *_X.npy / *_Y.npy files.")
    parser.add_argument("--logits-dir", type=Path, required=True, help="Directory containing fold*.npz saved by EdgeConv.")
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--model-name", type=str, default="edgeconv")
    parser.add_argument(
        "--aggregate",
        choices=["none", "subject"],
        default="none",
        help="Optional aggregation level for the exported prediction file.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_subjects(data_dir: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    subjects: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for x_path in sorted(data_dir.glob("*_X.npy")):
        y_path = x_path.with_name(x_path.name.replace("_X.npy", "_Y.npy"))
        if not y_path.exists():
            continue
        subject_id = x_path.name[:-6]
        x = np.load(x_path)
        y = np.load(y_path)
        subjects.append((subject_id, x, y))
    if not subjects:
        raise FileNotFoundError(f"No *_X.npy files found under {data_dir}")
    return subjects


def iter_window_metadata(
    subject_id: str,
    labels: np.ndarray,
    window_size: int,
    stride: int,
) -> List[Dict[str, object]]:
    labels = np.asarray(labels, dtype=np.int64)
    T = len(labels)

    if window_size <= 1:
        return [
            {
                "sample_id": f"{subject_id}|start={idx:04d}|ws=1|stride=1",
                "subject_id": subject_id,
                "group_id": f"{subject_id}|start={idx:04d}|ws=1|stride=1",
                "label": int(labels[idx]),
            }
            for idx in range(T)
        ]

    rows: List[Dict[str, object]] = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        window_label = int(np.bincount(labels[start:end]).argmax())
        chunk_id = f"{subject_id}|start={start:04d}|ws={window_size}|stride={stride}"
        rows.append(
            {
                "sample_id": chunk_id,
                "subject_id": subject_id,
                "group_id": chunk_id,
                "label": window_label,
            }
        )
    return rows


def parse_fold_number(path: Path) -> int:
    match = re.search(r"fold(\d+)", path.stem)
    if not match:
        raise ValueError(f"Unable to parse fold number from {path.name}")
    return int(match.group(1))


def main() -> None:
    args = parse_args()

    subjects = load_subjects(args.data_dir)
    subject_ids = [subject_id for subject_id, _, _ in subjects]

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_splits = list(enumerate(kf.split(np.arange(len(subjects))), start=1))

    rows: List[Dict[str, object]] = []
    fold_files = sorted(args.logits_dir.glob("fold*.npz"), key=parse_fold_number)
    if not fold_files:
        raise FileNotFoundError(f"No fold*.npz files found under {args.logits_dir}")

    split_lookup = {fold_idx: test_idx for fold_idx, (_, test_idx) in fold_splits}

    for fold_file in fold_files:
        fold_idx = parse_fold_number(fold_file)
        if fold_idx not in split_lookup:
            raise ValueError(f"Fold {fold_idx} not found in reconstructed KFold split.")

        payload = np.load(fold_file)
        logits = payload["logits"]
        labels = payload["labels"]
        probs = ensure_binary_probabilities(logits=logits)

        fold_meta: List[Dict[str, object]] = []
        for subject_index in split_lookup[fold_idx]:
            subject_id, _, y = subjects[subject_index]
            fold_meta.extend(iter_window_metadata(subject_id, y, args.window_size, args.stride))

        if len(fold_meta) != len(labels):
            raise ValueError(
                f"{fold_file.name}: reconstructed sample count {len(fold_meta)} "
                f"does not match saved labels {len(labels)}."
            )

        for idx, meta in enumerate(fold_meta):
            expected_label = int(meta["label"])
            saved_label = int(labels[idx])
            if expected_label != saved_label:
                raise ValueError(
                    f"{fold_file.name}: label mismatch at row {idx}: "
                    f"reconstructed={expected_label}, saved={saved_label}"
                )

            rows.append(
                {
                    "model_name": args.model_name,
                    "sample_id": meta["sample_id"],
                    "subject_id": meta["subject_id"],
                    "group_id": meta["group_id"],
                    "label": saved_label,
                    "prob_0": float(probs[idx, 0]),
                    "prob_1": float(probs[idx, 1]),
                }
            )

    if args.aggregate == "subject":
        rows = aggregate_rows(rows, key="subject_id")

    write_prediction_rows(args.output, rows)
    print(f"Exported {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()