from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


CH_NAMES_30 = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "FT7", "FC3", "FCZ", "FC4", "FT8",
    "T7", "C3", "CZ", "C4", "T8",
    "TP7", "CP3", "CPZ", "CP4", "TP8",
    "P7", "P3", "PZ", "P4", "P8",
    "O1", "OZ", "O2",
]


STANDARD_1020 = [
    "FP1", "FPZ", "FP2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2",
    "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
    "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
    "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
    "T1", "T2", "FTT9h", "TTP7h", "TPP9h", "FTT10h", "TPP8h", "TPP10h",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]


HEMISPHERIC_PAIRS = [
    (0, 1), (2, 6), (3, 5), (7, 11), (8, 10),
    (12, 16), (13, 15), (17, 21), (18, 20),
    (22, 26), (23, 25), (27, 29),
]


@dataclass
class ChunkRecord:
    subject_id: str
    chunk_id: str
    label: int
    start_idx: int
    end_idx: int


def add_asymmetry_channels(x: np.ndarray) -> np.ndarray:
    left_idx = [l for l, _ in HEMISPHERIC_PAIRS]
    right_idx = [r for _, r in HEMISPHERIC_PAIRS]
    asym = x[:, left_idx, :] - x[:, right_idx, :]
    return np.concatenate([x, asym], axis=1)


def zscore_subject(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


def find_npy_subjects(data_dir: str | Path) -> List[tuple[str, Path, Path]]:
    data_dir = Path(data_dir)
    subjects: List[tuple[str, Path, Path]] = []
    for x_file in sorted(data_dir.glob("*_X.npy")):
        y_file = data_dir / x_file.name.replace("_X.npy", "_Y.npy")
        if not y_file.exists():
            continue
        subject_id = x_file.stem.replace("timedata_X", "")
        subjects.append((subject_id, x_file, y_file))
    if not subjects:
        raise FileNotFoundError(f"No *_X.npy / *_Y.npy pairs found under {data_dir}")
    return subjects


def build_class_chunks_from_de(
    x: np.ndarray,
    y: np.ndarray,
    subject_id: str,
    window_size: int,
    stride: int,
    use_asymmetry: bool = False,
) -> tuple[np.ndarray, List[ChunkRecord]]:
    features: List[np.ndarray] = []
    records: List[ChunkRecord] = []

    for label in sorted(np.unique(y).tolist()):
        label = int(label)
        idx = np.where(y == label)[0]
        x_label = x[idx]
        for start in range(0, len(x_label) - window_size + 1, stride):
            end = start + window_size
            chunk = x_label[start:end]
            chunk = chunk.transpose(1, 0, 2).reshape(30, -1)
            chunk = chunk[np.newaxis, ...]
            if use_asymmetry:
                chunk = add_asymmetry_channels(chunk)
            chunk = chunk[0]
            chunk_id = f"{subject_id}|{label}|{start}|{window_size}"
            features.append(chunk.astype(np.float32))
            records.append(
                ChunkRecord(
                    subject_id=subject_id,
                    chunk_id=chunk_id,
                    label=label,
                    start_idx=start,
                    end_idx=end,
                )
            )

    if not features:
        raise ValueError(f"No chunks built for subject {subject_id}. Check window_size/stride.")

    return np.stack(features, axis=0), records


def build_input_chans(ch_names: Sequence[str]) -> List[int]:
    return [0] + [STANDARD_1020.index(ch_name) + 1 for ch_name in ch_names]


def load_checkpoint_state_dict(path: str | Path, key_candidates: Sequence[str] | None = None) -> Dict[str, torch.Tensor]:
    load_kwargs = {"map_location": "cpu"}
    try:
        checkpoint = torch.load(str(path), weights_only=False, **load_kwargs)
    except TypeError:
        # Older PyTorch versions do not accept weights_only.
        checkpoint = torch.load(str(path), **load_kwargs)

    if isinstance(checkpoint, dict):
        key_candidates = list(key_candidates or []) + ["state_dict", "model_state", "model", "module"]
        for key in key_candidates:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        clean_key = key[7:] if key.startswith("module.") else key
        state_dict[clean_key] = value
    return state_dict


def batched_softmax_predict(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    forward_fn=None,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            batch_np = x[start:end]
            try:
                batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
            except RuntimeError as exc:
                if "Numpy is not available" not in str(exc) and "_ARRAY_API" not in str(exc):
                    raise
                # Fall back to Python lists when PyTorch cannot use the NumPy bridge.
                batch = torch.tensor(batch_np.tolist(), dtype=torch.float32, device=device)
            logits = forward_fn(model, batch) if forward_fn is not None else model(batch)
            if isinstance(logits, tuple):
                logits = logits[-1]
            probs_tensor = torch.softmax(logits, dim=-1).detach().cpu()
            try:
                probs = probs_tensor.numpy()
            except RuntimeError as exc:
                if "Numpy is not available" not in str(exc) and "_ARRAY_API" not in str(exc):
                    raise
                probs = np.asarray(probs_tensor.tolist(), dtype=np.float32)
            outputs.append(probs)
    return np.concatenate(outputs, axis=0)


def write_prediction_csv(
    out_csv: str | Path,
    records: Sequence[ChunkRecord],
    probs: np.ndarray,
    source: str,
) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject_id",
            "chunk_id",
            "start_idx",
            "end_idx",
            "label",
            "prob_0",
            "prob_1",
            "pred_class",
            "source",
        ])
        for rec, prob in zip(records, probs):
            writer.writerow([
                rec.subject_id,
                rec.chunk_id,
                rec.start_idx,
                rec.end_idx,
                rec.label,
                float(prob[0]),
                float(prob[1]),
                int(np.argmax(prob)),
                source,
            ])


def summarize_predictions(source: str, records: Sequence[ChunkRecord], probs: np.ndarray) -> str:
    labels = np.asarray([rec.label for rec in records], dtype=np.int64)
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == labels))
    return f"{source}: exported {len(records)} chunks, accuracy={acc * 100:.2f}%"
