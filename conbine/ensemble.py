from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class PredictionTable:
    name: str
    keys: List[str]
    labels: np.ndarray
    probs: np.ndarray


def load_prediction_csv(path: Path, name: str | None = None) -> PredictionTable:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    key_fields = ["subject_id", "chunk_id", "start_idx", "end_idx"]
    active_key_fields = [field for field in key_fields if field in rows[0] and row_has_value(rows, field)]
    if not active_key_fields:
        if "subject_id" in rows[0]:
            active_key_fields = ["subject_id"]
        else:
            raise ValueError(f"{path} must contain at least subject_id")

    keys: List[str] = []
    labels: List[int] = []
    probs: List[Tuple[float, float]] = []

    for row in rows:
        key = "|".join(str(row.get(field, "")).strip() for field in active_key_fields)
        p0 = float(row["prob_0"])
        p1 = float(row["prob_1"])
        total = p0 + p1
        if total <= 0:
            raise ValueError(f"Invalid probability row in {path}: {row}")
        p0 /= total
        p1 /= total
        keys.append(key)
        labels.append(int(row["label"]))
        probs.append((p0, p1))

    return PredictionTable(
        name=name or path.stem,
        keys=keys,
        labels=np.asarray(labels, dtype=np.int64),
        probs=np.asarray(probs, dtype=np.float64),
    )


def row_has_value(rows: Sequence[Dict[str, str]], field: str) -> bool:
    for row in rows:
        if str(row.get(field, "")).strip():
            return True
    return False


def align_tables(tables: Sequence[PredictionTable]) -> Tuple[List[str], np.ndarray, List[np.ndarray]]:
    if len(tables) < 2:
        raise ValueError("Need at least two prediction tables for ensemble")

    shared_keys = set(tables[0].keys)
    for table in tables[1:]:
        shared_keys &= set(table.keys)

    if not shared_keys:
        raise ValueError("No overlapping sample keys found across prediction files")

    ordered_keys = [key for key in tables[0].keys if key in shared_keys]
    key_to_pos = [{key: idx for idx, key in enumerate(table.keys)} for table in tables]

    aligned_labels = tables[0].labels[[key_to_pos[0][key] for key in ordered_keys]]
    aligned_probs: List[np.ndarray] = []
    for table_idx, table in enumerate(tables):
        idxs = [key_to_pos[table_idx][key] for key in ordered_keys]
        labels = table.labels[idxs]
        if not np.array_equal(labels, aligned_labels):
            raise ValueError(f"Label mismatch after alignment for model {table.name}")
        aligned_probs.append(table.probs[idxs])

    return ordered_keys, aligned_labels, aligned_probs


def accuracy_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred == labels))


def log_loss_from_probs(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> float:
    probs = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels])))


def weighted_average(probs_list: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    weights_arr = np.asarray(weights, dtype=np.float64)
    if weights_arr.ndim != 1 or len(weights_arr) != len(probs_list):
        raise ValueError("weights length must match number of models")
    if np.allclose(weights_arr.sum(), 0.0):
        raise ValueError("weights sum cannot be zero")
    weights_arr = weights_arr / weights_arr.sum()
    out = np.zeros_like(probs_list[0], dtype=np.float64)
    for weight, probs in zip(weights_arr, probs_list):
        out += weight * probs
    out /= np.sum(out, axis=1, keepdims=True)
    return out


def auto_weights_from_accuracy(probs_list: Sequence[np.ndarray], labels: np.ndarray) -> np.ndarray:
    accs = np.asarray([accuracy_from_probs(probs, labels) for probs in probs_list], dtype=np.float64)
    shifted = np.clip(accs - 0.5, 1e-4, None)
    return shifted / shifted.sum()


def orthogonalize_features(x: np.ndarray) -> np.ndarray:
    cols: List[np.ndarray] = []
    for col_idx in range(x.shape[1]):
        v = x[:, col_idx].copy()
        for basis in cols:
            denom = np.dot(basis, basis)
            if denom > 1e-12:
                v = v - basis * (np.dot(v, basis) / denom)
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            v = v / norm
        cols.append(v)
    return np.stack(cols, axis=1)


def solve_ridge_binary(x: np.ndarray, y: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    eye = np.eye(x_aug.shape[1], dtype=x.dtype)
    eye[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + reg * eye
    rhs = x_aug.T @ y
    return np.linalg.solve(lhs, rhs)


def orthogonal_stacking(probs_list: Sequence[np.ndarray], labels: np.ndarray, reg: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    features = np.concatenate([probs[:, 1:2] for probs in probs_list], axis=1)
    centered = features - np.mean(features, axis=0, keepdims=True)
    ortho = orthogonalize_features(centered)
    coef = solve_ridge_binary(ortho, labels.astype(np.float64), reg=reg)
    x_aug = np.concatenate([ortho, np.ones((ortho.shape[0], 1), dtype=ortho.dtype)], axis=1)
    score = x_aug @ coef
    prob1 = np.clip(score, 0.0, 1.0)
    probs = np.stack([1.0 - prob1, prob1], axis=1)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs, coef


def write_predictions(
    out_path: Path,
    keys: Sequence[str],
    labels: np.ndarray,
    probs: np.ndarray,
    source: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "label", "prob_0", "prob_1", "pred_class", "source"])
        for key, label, prob in zip(keys, labels, probs):
            subject_id = key.split("|")[0]
            writer.writerow([
                subject_id,
                int(label),
                float(prob[0]),
                float(prob[1]),
                int(np.argmax(prob)),
                source,
            ])


def print_metrics(name: str, probs: np.ndarray, labels: np.ndarray) -> None:
    acc = accuracy_from_probs(probs, labels)
    loss = log_loss_from_probs(probs, labels)
    print(f"{name}: accuracy={acc * 100:.2f}%  logloss={loss:.5f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble already-exported model predictions without modifying original source code.")
    p.add_argument("--pred_csv", nargs="+", required=True, help="Prediction CSV files from different models.")
    p.add_argument("--names", nargs="*", default=None, help="Optional names aligned with --pred_csv.")
    p.add_argument("--mode", choices=["soft", "orthogonal", "both"], default="both")
    p.add_argument("--weights", nargs="*", type=float, default=None, help="Manual weights for soft voting.")
    p.add_argument("--auto_weight", action="store_true", help="Use validation accuracy-derived weights for soft voting.")
    p.add_argument("--reg", type=float, default=1e-3, help="Ridge regularization for orthogonal stacking.")
    p.add_argument("--out_dir", type=str, default="conbine/output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_paths = [Path(p) for p in args.pred_csv]
    if args.names and len(args.names) != len(pred_paths):
        raise ValueError("--names length must match --pred_csv length")

    tables = [
        load_prediction_csv(path, None if args.names is None else args.names[idx])
        for idx, path in enumerate(pred_paths)
    ]

    keys, labels, probs_list = align_tables(tables)

    for table, probs in zip(tables, probs_list):
        print_metrics(table.name, probs, labels)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"soft", "both"}:
        if args.weights is not None:
            weights = np.asarray(args.weights, dtype=np.float64)
        elif args.auto_weight:
            weights = auto_weights_from_accuracy(probs_list, labels)
        else:
            weights = np.ones(len(probs_list), dtype=np.float64) / len(probs_list)
        soft_probs = weighted_average(probs_list, weights)
        print(f"soft_weights={weights.tolist()}")
        print_metrics("soft_voting", soft_probs, labels)
        write_predictions(out_dir / "soft_voting_predictions.csv", keys, labels, soft_probs, "soft_voting")

    if args.mode in {"orthogonal", "both"}:
        ortho_probs, coef = orthogonal_stacking(probs_list, labels, reg=args.reg)
        print(f"orthogonal_coef={coef.tolist()}")
        print_metrics("orthogonal_stacking", ortho_probs, labels)
        write_predictions(out_dir / "orthogonal_stacking_predictions.csv", keys, labels, ortho_probs, "orthogonal_stacking")


if __name__ == "__main__":
    main()
