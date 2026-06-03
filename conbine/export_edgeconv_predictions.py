from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
EDGE_DIR = ROOT / "code" / "EdgeConv"
if str(EDGE_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_DIR))

from model import EdgeDGCNN  # type: ignore
from spatial_prior import get_spatial_dist  # type: ignore

from common import (
    batched_softmax_predict,
    build_class_chunks_from_de,
    find_npy_subjects,
    load_checkpoint_state_dict,
    summarize_predictions,
    write_prediction_csv,
    zscore_subject,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EdgeConv predictions into the unified conbine CSV format.")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing *_X.npy / *_Y.npy.")
    p.add_argument("--checkpoint", type=str, required=True, help="EdgeConv checkpoint path.")
    p.add_argument("--out_csv", type=str, default="conbine/output/edgeconv_predictions.csv")
    p.add_argument("--window_size", type=int, default=40)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--k_neighbors", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_asymmetry", action="store_true", default=False)
    p.add_argument("--use_band_se", action="store_true", default=False)
    return p.parse_args()


def infer_dimensions_from_state_dict(state_dict: dict[str, torch.Tensor], fallback_nodes: int) -> tuple[int, int]:
    fc1_weight = state_dict["fc1.linear.weight"]
    flat_dim = int(fc1_weight.shape[1])
    num_nodes = fallback_nodes
    if flat_dim % num_nodes != 0:
        raise ValueError(f"Cannot infer in_features from flat_dim={flat_dim}, num_nodes={num_nodes}")
    hidden_dim = flat_dim // num_nodes
    if hidden_dim != 64:
        raise ValueError(f"Unexpected hidden dim {hidden_dim}, exporter currently expects EdgeConv hidden_dim=64")

    bn_input_weight = state_dict["bn_input.weight"]
    in_features = int(bn_input_weight.shape[0])
    return in_features, num_nodes


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    subjects = find_npy_subjects(args.data_dir)
    state_dict = load_checkpoint_state_dict(args.checkpoint)
    num_nodes = 42 if args.use_asymmetry else 30
    in_features, num_nodes = infer_dimensions_from_state_dict(state_dict, num_nodes)

    spatial_dist = None if args.use_asymmetry else get_spatial_dist(device=device)
    model = EdgeDGCNN(
        in_features=in_features,
        num_nodes=num_nodes,
        k=args.k_neighbors,
        nclass=2,
        spatial_dist=spatial_dist,
        use_supcon=False,
        use_band_se=args.use_band_se,
        n_bands=max(1, in_features // max(args.window_size, 1)),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)

    all_records = []
    all_probs = []

    for subject_id, x_file, y_file in subjects:
        x = np.load(x_file)
        y = np.load(y_file)
        x = zscore_subject(x)
        features, records = build_class_chunks_from_de(
            x,
            y,
            subject_id=subject_id,
            window_size=args.window_size,
            stride=args.stride,
            use_asymmetry=args.use_asymmetry,
        )
        probs = batched_softmax_predict(model, features, device, args.batch_size)
        all_records.extend(records)
        all_probs.append(probs)

    probs_all = np.concatenate(all_probs, axis=0)
    write_prediction_csv(args.out_csv, all_records, probs_all, "edgeconv")
    print(summarize_predictions("edgeconv", all_records, probs_all))


if __name__ == "__main__":
    main()
