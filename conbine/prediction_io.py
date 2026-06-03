from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

FIELDNAMES = [
    "model_name",
    "sample_id",
    "subject_id",
    "group_id",
    "label",
    "prob_0",
    "prob_1",
]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def ensure_binary_probabilities(
    logits: np.ndarray | None = None,
    probs: np.ndarray | None = None,
) -> np.ndarray:
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = np.stack([1.0 - probs, probs], axis=1)
        if probs.shape[1] != 2:
            raise ValueError(f"Expected binary probabilities, got shape {probs.shape}")
        return probs

    if logits is None:
        raise ValueError("Either logits or probs must be provided")

    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        prob_1 = sigmoid(logits)
        return np.stack([1.0 - prob_1, prob_1], axis=1)
    if logits.shape[1] == 1:
        prob_1 = sigmoid(logits[:, 0])
        return np.stack([1.0 - prob_1, prob_1], axis=1)
    if logits.shape[1] == 2:
        return softmax(logits)
    raise ValueError(f"Expected binary logits, got shape {logits.shape}")


def write_prediction_rows(path: str | Path, rows: Sequence[Dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            normalized = {key: row.get(key, "") for key in FIELDNAMES}
            writer.writerow(normalized)


def read_prediction_rows(path: str | Path) -> List[Dict[str, object]]:
    path = Path(path)
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            row["prob_0"] = float(row["prob_0"])
            row["prob_1"] = float(row["prob_1"])
            rows.append(row)
    return rows


def resolve_key(row: Dict[str, object], key: str) -> str:
    value = str(row.get(key, "") or "")
    if value:
        return value

    fallbacks = ["group_id", "sample_id", "subject_id"]
    for fallback in fallbacks:
        value = str(row.get(fallback, "") or "")
        if value:
            return value

    raise KeyError(f"Unable to resolve key '{key}' for row: {row}")


def aggregate_rows(rows: Sequence[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)