from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from einops import rearrange
from timm.models import create_model

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
LABRAM_DIR = ROOT / "code" / "LaBraM"
if str(LABRAM_DIR) not in sys.path:
    sys.path.insert(0, str(LABRAM_DIR))

import modeling_finetune  # type: ignore

from common import (
    CH_NAMES_30,
    batched_softmax_predict,
    build_class_chunks_from_de,
    build_input_chans,
    find_npy_subjects,
    load_checkpoint_state_dict,
    summarize_predictions,
    write_prediction_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export LaBraM predictions into the unified conbine CSV format.")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing *_X.npy / *_Y.npy.")
    p.add_argument("--checkpoint", type=str, default=str(ROOT / "code" / "LaBraM" / "checkpoints" / "labram-base.pth"))
    p.add_argument("--out_csv", type=str, default="conbine/output/labram_predictions.csv")
    p.add_argument("--window_size", type=int, default=40)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model", type=str, default="labram_base_patch200_200")
    return p.parse_args()


def forward_labram(model: torch.nn.Module, batch: torch.Tensor, input_chans: List[int]) -> torch.Tensor:
    batch = batch / 100.0
    batch = rearrange(batch, "B N (A T) -> B N A T", T=200)
    return model(batch, input_chans=input_chans)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    subjects = find_npy_subjects(args.data_dir)
    input_chans = build_input_chans(CH_NAMES_30)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=2,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=True,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=True,
    ).to(device)

    state_dict = load_checkpoint_state_dict(args.checkpoint, key_candidates=["student"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"Warning: unexpected keys ignored: {unexpected}")
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")

    all_records = []
    all_probs = []

    for subject_id, x_file, y_file in subjects:
        x = np.load(x_file).astype(np.float32)
        y = np.load(y_file).astype(np.int64)
        features, records = build_class_chunks_from_de(
            x,
            y,
            subject_id=subject_id,
            window_size=args.window_size,
            stride=args.stride,
            use_asymmetry=False,
        )
        probs = batched_softmax_predict(
            model,
            features,
            device,
            args.batch_size,
            forward_fn=lambda m, batch: forward_labram(m, batch, input_chans),
        )
        all_records.extend(records)
        all_probs.append(probs)

    probs_all = np.concatenate(all_probs, axis=0)
    write_prediction_csv(args.out_csv, all_records, probs_all, "labram")
    print(summarize_predictions("labram", all_records, probs_all))


if __name__ == "__main__":
    main()
