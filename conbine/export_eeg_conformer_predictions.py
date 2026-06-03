from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import scipy.io as sio
import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
EEG_CONF_DIR = ROOT / "EEG-Conformer"
if str(EEG_CONF_DIR) not in sys.path:
    sys.path.insert(0, str(EEG_CONF_DIR))

from conformer import ExGAN, ViT  # type: ignore

from common import ChunkRecord, batched_softmax_predict, load_checkpoint_state_dict, summarize_predictions, write_prediction_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EEG-Conformer predictions into the unified conbine CSV format.")
    p.add_argument("--data_dir", type=str, default=str(ROOT / "EEG-Conformer" / "data" / "processed_normal"))
    p.add_argument("--checkpoint", type=str, default=str(ROOT / "EEG-Conformer" / "last_params" / "better_D2_H4_S40_best1.pth"))
    p.add_argument("--out_csv", type=str, default="conbine/output/eeg_conformer_predictions.csv")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--emb_size", type=int, default=40)
    p.add_argument("--depth", type=int, default=2)
    return p.parse_args()


def subject_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**9, path.stem)


def load_subject_windows(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(str(mat_path))
    data = np.ascontiguousarray(mat["data"], dtype=np.float32)
    labels = np.ascontiguousarray(mat["label"].flatten(), dtype=np.int64)
    return data, labels


def subject_train_stats(data: np.ndarray, labels: np.ndarray, subject_seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx_list: List[np.ndarray] = []
    for cls in [0, 1]:
        cls_idx = np.where(labels == cls)[0]
        rng = np.random.RandomState(subject_seed)
        cls_idx = cls_idx[rng.permutation(len(cls_idx))]
        split_point = int(len(cls_idx) * 0.8)
        train_idx_list.append(cls_idx[:split_point])
    train_idx = np.concatenate(train_idx_list)
    train_data = data[train_idx]
    mu = train_data.mean(axis=(0, 2), keepdims=True)
    std = train_data.std(axis=(0, 2), keepdims=True) + 1e-8
    return mu, std


def build_records(subject_id: str, labels: np.ndarray) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    class_counter = {0: 0, 1: 0}
    for idx, label in enumerate(labels.tolist()):
        label = int(label)
        class_index = class_counter[label]
        class_counter[label] += 1
        chunk_id = f"{subject_id}|{label}|{class_index}|1"
        records.append(
            ChunkRecord(
                subject_id=subject_id,
                chunk_id=chunk_id,
                label=label,
                start_idx=class_index,
                end_idx=class_index + 1,
            )
        )
    return records


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=250, emb_size=args.emb_size)
    model = ViT(
        emb_size=args.emb_size,
        depth=args.depth,
        n_classes=2,
        n_channels=30,
        seq_len=seq_len,
    ).to(device)

    state_dict = load_checkpoint_state_dict(args.checkpoint)
    model.load_state_dict(state_dict, strict=False)

    mat_files = sorted(Path(args.data_dir).glob("*.mat"), key=subject_sort_key)
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {args.data_dir}")

    all_records = []
    all_probs = []

    for mat_path in mat_files:
        subject_match = re.search(r"(HC\d+)", mat_path.stem.upper())
        if not subject_match:
            continue
        subject_id = subject_match.group(1)
        data, labels = load_subject_windows(mat_path)
        subject_seed_match = re.search(r"(\d+)", subject_id)
        subject_seed = int(subject_seed_match.group(1)) if subject_seed_match else 42
        mu, std = subject_train_stats(data, labels, subject_seed)
        data = (data - mu) / std
        data = np.ascontiguousarray(data[:, np.newaxis, :, :], dtype=np.float32)
        records = build_records(subject_id, labels)
        probs = batched_softmax_predict(model, data, device, args.batch_size)
        all_records.extend(records)
        all_probs.append(probs)

    probs_all = np.concatenate(all_probs, axis=0)
    write_prediction_csv(args.out_csv, all_records, probs_all, "eeg_conformer")
    print(summarize_predictions("eeg_conformer", all_records, probs_all))


if __name__ == "__main__":
    main()
