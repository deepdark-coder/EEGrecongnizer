import argparse
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
TRIALS_PER_SUBJECT = 8

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple late fusion for EdgeConv, LaBraM and EEG-Conformer."
    )
    parser.add_argument("--de_data_dir", type=str, default=str(ROOT / "data" / "code" / "processed_testset"))
    parser.add_argument("--mat_data_dir", type=str, default=str(ROOT / "EEG-Conformer" / "data" / "processed_testset"))
    parser.add_argument(
        "--extra_data_dir",
        type=str,
        nargs="*",
        default=[],
        help="Optional extra data directories. Any directory containing *_X.npy will be added to the DE-based models, and any directory containing *.mat will be added to EEG-Conformer.",
    )

    parser.add_argument("--edge_ckpt", type=str, default=str(ROOT / "params" / "edgeconv_best.pth"))
    parser.add_argument("--labram_ckpt", type=str, default=str(ROOT / "params" / "labram_best.pth"))
    parser.add_argument("--eeg_ckpt", type=str, default=str(ROOT / "EEG-Conformer" / "last_params" / "raw_D2_H4_S24_best1.pth"))

    parser.add_argument("--edge_weight", type=float, default=1.0)
    parser.add_argument("--labram_weight", type=float, default=1.0)
    parser.add_argument("--eeg_weight", type=float, default=0.1)

    parser.add_argument("--edge_window", type=int, default=40)
    parser.add_argument("--edge_stride", type=int, default=1)
    parser.add_argument("--labram_window", type=int, default=40)
    parser.add_argument("--labram_stride", type=int, default=5)

    parser.add_argument("--edge_k", type=int, default=20)
    parser.add_argument("--edge_use_asymmetry",default=False, action="store_true")
    parser.add_argument("--edge_use_band_se",default=False, action="store_true")

    parser.add_argument("--eeg_emb_size", type=int, default=24)
    parser.add_argument("--eeg_depth", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_csv", type=str, default="infer/output/fused_subject_label_scores.csv")
    return parser.parse_args()


def load_checkpoint_state_dict(path, extra_keys=None):
    load_kwargs = {"map_location": "cpu"}
    try:
        checkpoint = torch.load(str(path), weights_only=False, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(str(path), **load_kwargs)

    if isinstance(checkpoint, dict):
        keys = []
        if extra_keys:
            keys.extend(extra_keys)
        keys.extend(["state_dict", "model_state", "model", "module"])
        for key in keys:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: %s" % path)

    state_dict = {}
    for key, value in checkpoint.items():
        clean_key = key[7:] if key.startswith("module.") else key
        state_dict[clean_key] = value
    return state_dict


def infer_conformer_raw_config(state_dict, fallback_emb_size, fallback_depth):
    emb_size = int(fallback_emb_size)
    proj_weight = state_dict.get("patch_embedding.projection.0.weight")
    if proj_weight is not None and getattr(proj_weight, "ndim", 0) >= 1:
        emb_size = int(proj_weight.shape[0])

    depth_indices = []
    for key in state_dict:
        if not key.startswith("transformer."):
            continue
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            depth_indices.append(int(parts[1]))
    depth = max(depth_indices) + 1 if depth_indices else int(fallback_depth)

    seq_len = None
    cls_weight = state_dict.get("cls_head.fc.0.weight")
    if cls_weight is not None and getattr(cls_weight, "ndim", 0) >= 2 and emb_size > 0:
        flattened_dim = int(cls_weight.shape[1])
        if flattened_dim % emb_size == 0:
            seq_len = flattened_dim // emb_size
    return emb_size, depth, seq_len


def safe_tensor(batch_np, device):
    try:
        return torch.tensor(batch_np, dtype=torch.float32, device=device)
    except RuntimeError as exc:
        message = str(exc)
        if "Numpy is not available" not in message and "_ARRAY_API" not in message:
            raise
        return torch.tensor(batch_np.tolist(), dtype=torch.float32, device=device)


def safe_numpy(tensor):
    tensor = tensor.detach().cpu()
    try:
        return tensor.numpy()
    except RuntimeError as exc:
        message = str(exc)
        if "Numpy is not available" not in message and "_ARRAY_API" not in message:
            raise
        return np.asarray(tensor.tolist(), dtype=np.float32)


def predict_probs(model, x, device, batch_size, forward_fn=None):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            batch = safe_tensor(x[start:end], device)
            if forward_fn is None:
                logits = model(batch)
            else:
                logits = forward_fn(model, batch)
            if isinstance(logits, tuple):
                logits = logits[-1]
            probs = torch.softmax(logits, dim=-1)
            outputs.append(safe_numpy(probs))
    return np.concatenate(outputs, axis=0)


def zscore_subject(x):
    x = x.astype(np.float64, copy=False)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


def add_asymmetry_channels(x):
    left_idx = [left for left, _ in HEMISPHERIC_PAIRS]
    right_idx = [right for _, right in HEMISPHERIC_PAIRS]
    asym = x[:, left_idx, :] - x[:, right_idx, :]
    return np.concatenate([x, asym], axis=1)


def parse_npy_subject_id(x_file):
    subject_id = x_file.stem.replace("timedata_X", "")
    subject_id = subject_id.strip("_")
    if not subject_id:
        subject_id = x_file.stem
    return subject_id


def parse_mat_subject_id(mat_file):
    match = re.search(r"(HC\d+)", mat_file.stem.upper())
    if match:
        return match.group(1)
    return mat_file.stem


def normalize_input_dirs(primary_dir, extra_dirs):
    dirs = []
    seen = set()
    for raw_path in [primary_dir] + list(extra_dirs):
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError("Data directory does not exist: %s" % path)
        if not path.is_dir():
            raise NotADirectoryError("Expected a directory path: %s" % path)
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        dirs.append(path)
    return dirs


def find_de_subjects(primary_dir, extra_dirs):
    subjects = []
    discovered_dirs = []
    subject_sources = {}
    for data_dir in normalize_input_dirs(primary_dir, extra_dirs):
        discovered_dirs.append(str(data_dir))
        for x_file in sorted(data_dir.glob("*_X.npy")):
            y_file = data_dir / x_file.name.replace("_X.npy", "_Y.npy")
            if not y_file.exists():
                continue
            subject_id = parse_npy_subject_id(x_file)
            if subject_id in subject_sources:
                raise ValueError(
                    "Duplicate DE subject_id %s found in %s and %s"
                    % (subject_id, subject_sources[subject_id], x_file)
                )
            subject_sources[subject_id] = str(x_file)
            subjects.append((subject_id, x_file, y_file))
    if not subjects:
        raise FileNotFoundError("No *_X.npy / *_Y.npy pairs found under: %s" % ", ".join(discovered_dirs))
    return sorted(subjects, key=lambda item: item[0])


def find_mat_files(primary_dir, extra_dirs):
    mat_files = []
    discovered_dirs = []
    subject_sources = {}
    for data_dir in normalize_input_dirs(primary_dir, extra_dirs):
        discovered_dirs.append(str(data_dir))
        for mat_path in sorted(data_dir.glob("*.mat"), key=sort_mat_key):
            subject_id = parse_mat_subject_id(mat_path)
            if subject_id in subject_sources:
                raise ValueError(
                    "Duplicate EEG subject_id %s found in %s and %s"
                    % (subject_id, subject_sources[subject_id], mat_path)
                )
            subject_sources[subject_id] = str(mat_path)
            mat_files.append(mat_path)
    if not mat_files:
        raise FileNotFoundError("No .mat files found under: %s" % ", ".join(discovered_dirs))
    return sorted(mat_files, key=sort_mat_key)


def build_de_chunks(x_label, window_size, stride, use_asymmetry):
    features = []
    if len(x_label) < window_size:
        return None
    for start in range(0, len(x_label) - window_size + 1, stride):
        end = start + window_size
        chunk = x_label[start:end]
        chunk = chunk.transpose(1, 0, 2).reshape(30, -1)
        chunk = chunk[np.newaxis, ...]
        if use_asymmetry:
            chunk = add_asymmetry_channels(chunk)
        features.append(chunk[0].astype(np.float32))
    if not features:
        return None
    return np.stack(features, axis=0)


def build_input_chans():
    return [0] + [STANDARD_1020.index(ch_name) + 1 for ch_name in CH_NAMES_30]


def sort_mat_key(path):
    match = re.search(r"(\d+)", path.stem)
    if match:
        return (int(match.group(1)), path.stem)
    return (10**9, path.stem)


def build_trial_slices(length, num_trials=TRIALS_PER_SUBJECT):
    if length < num_trials:
        raise ValueError(
            "Expected at least %s samples/windows for trial splitting, got %s"
            % (num_trials, length)
        )
    base = length // num_trials
    remainder = length % num_trials
    slices = []
    start = 0
    for trial_id in range(1, num_trials + 1):
        step = base + (1 if trial_id <= remainder else 0)
        end = start + step
        slices.append((trial_id, start, end))
        start = end
    return slices


def majority_label(labels):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        raise ValueError("Cannot compute majority label from an empty label array.")
    counts = np.bincount(labels)
    return int(counts.argmax())


def merge_label_maps(base_map, new_map, source_name):
    for key, value in new_map.items():
        value = int(value)
        if key in base_map and int(base_map[key]) != value:
            print(
                "Warning: conflicting trial labels for %s from %s. Keeping existing=%s, new=%s"
                % (key, source_name, base_map[key], value)
            )
            continue
        base_map[key] = value


def subject_trial_sort_key(key):
    subject_id, trial_id = key
    match = re.search(r"(\d+)", str(subject_id))
    subject_num = int(match.group(1)) if match else 10**9
    return (subject_num, str(subject_id), int(trial_id))


def subject_train_stats(data, labels, subject_seed):
    train_idx_list = []
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


def load_edge_model(args, device):
    edge_dir = ROOT / "code" / "EdgeConv"
    if str(edge_dir) not in sys.path:
        sys.path.insert(0, str(edge_dir))

    from model import EdgeDGCNN
    from spatial_prior import get_spatial_dist

    state_dict = load_checkpoint_state_dict(args.edge_ckpt)
    num_nodes = 42 if args.edge_use_asymmetry else 30
    flat_dim = int(state_dict["fc1.linear.weight"].shape[1])
    in_features = int(state_dict["bn_input.weight"].shape[0])
    hidden_dim = flat_dim // num_nodes
    if hidden_dim != 64:
        raise ValueError("Unexpected EdgeConv hidden dim: %s" % hidden_dim)

    spatial_dist = None
    if not args.edge_use_asymmetry:
        spatial_dist = get_spatial_dist(device=device)

    model = EdgeDGCNN(
        in_features=in_features,
        num_nodes=num_nodes,
        k=args.edge_k,
        nclass=2,
        spatial_dist=spatial_dist,
        use_supcon=False,
        use_band_se=args.edge_use_band_se,
        n_bands=max(1, in_features // max(args.edge_window, 1)),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model


def run_edge_model(args, model, device, de_subjects):
    scores = {}
    label_map = {}
    for subject_id, x_file, y_file in de_subjects:
        x = np.load(x_file)
        y = np.load(y_file).astype(np.int64)
        x = zscore_subject(x)
        for trial_id, start, end in build_trial_slices(len(x)):
            features = build_de_chunks(
                x[start:end],
                args.edge_window,
                args.edge_stride,
                args.edge_use_asymmetry,
            )
            if features is None:
                continue
            probs = predict_probs(model, features, device, args.batch_size)
            scores[(subject_id, trial_id)] = float(probs[:, 1].mean())
            label_map[(subject_id, trial_id)] = majority_label(y[start:end])
    return scores, label_map


def load_labram_model(args, device):
    labram_dir = ROOT / "code" / "LaBraM"
    if str(labram_dir) not in sys.path:
        sys.path.insert(0, str(labram_dir))

    try:
        import modeling_finetune
    except ImportError as exc:
        raise ImportError("LaBraM needs timm. Please install timm before using --labram_ckpt.") from exc

    model = modeling_finetune.labram_base_patch200_200(
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

    state_dict = load_checkpoint_state_dict(args.labram_ckpt, extra_keys=["student"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print("LaBraM unexpected keys ignored:", unexpected)
    if missing:
        print("LaBraM missing keys:", missing)
    return model, build_input_chans()


def forward_labram(model, batch, input_chans):
    batch = batch / 100.0
    if batch.shape[-1] % 200 != 0:
        raise ValueError("LaBraM input width must be divisible by 200. Current width: %s" % batch.shape[-1])
    batch = batch.reshape(batch.shape[0], batch.shape[1], batch.shape[2] // 200, 200)
    return model(batch, input_chans=input_chans)


def run_labram_model(args, model, input_chans, device, de_subjects):
    scores = {}
    label_map = {}
    for subject_id, x_file, y_file in de_subjects:
        x = np.load(x_file).astype(np.float32)
        y = np.load(y_file).astype(np.int64)
        for trial_id, start, end in build_trial_slices(len(x)):
            features = build_de_chunks(
                x[start:end],
                args.labram_window,
                args.labram_stride,
                False,
            )
            if features is None:
                continue
            probs = predict_probs(
                model,
                features,
                device,
                args.batch_size,
                forward_fn=lambda current_model, batch: forward_labram(current_model, batch, input_chans),
            )
            scores[(subject_id, trial_id)] = float(probs[:, 1].mean())
            label_map[(subject_id, trial_id)] = majority_label(y[start:end])
    return scores, label_map


def load_eeg_model(args, device):
    eeg_dir = ROOT / "EEG-Conformer"
    if str(eeg_dir) not in sys.path:
        sys.path.insert(0, str(eeg_dir))

    from conformer_raw import ExGAN, ViT

    state_dict = load_checkpoint_state_dict(args.eeg_ckpt)
    emb_size, depth, seq_len = infer_conformer_raw_config(
        state_dict,
        fallback_emb_size=args.eeg_emb_size,
        fallback_depth=args.eeg_depth,
    )
    if seq_len is None:
        seq_len = ExGAN.get_seq_len(n_channels=30, n_times=250, emb_size=emb_size)
    print(
        "EEG-Conformer raw config: emb_size=%s depth=%s seq_len=%s"
        % (emb_size, depth, seq_len)
    )
    model = ViT(
        emb_size=emb_size,
        depth=depth,
        n_classes=2,
        n_channels=30,
        seq_len=seq_len,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print("EEG-Conformer unexpected keys ignored:", unexpected)
    if missing:
        print("EEG-Conformer missing keys:", missing)
    return model


def run_eeg_model(args, model, device, mat_files):
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError("EEG-Conformer inference needs scipy for reading .mat files.") from exc

    scores = {}
    label_map = {}
    for mat_path in mat_files:
        subject_id = parse_mat_subject_id(mat_path)
        mat = sio.loadmat(str(mat_path))
        data = np.ascontiguousarray(mat["data"], dtype=np.float32)
        labels = None
        if "label" in mat:
            labels = np.ascontiguousarray(mat["label"].flatten(), dtype=np.int64)

        if labels is not None:
            subject_seed_match = re.search(r"(\d+)", subject_id)
            subject_seed = int(subject_seed_match.group(1)) if subject_seed_match else 42
            mu, std = subject_train_stats(data, labels, subject_seed)
        else:
            mu = data.mean(axis=(0, 2), keepdims=True)
            std = data.std(axis=(0, 2), keepdims=True) + 1e-8
        data = (data - mu) / std
        data = np.ascontiguousarray(data[:, np.newaxis, :, :], dtype=np.float32)

        probs = predict_probs(model, data, device, args.batch_size)
        for trial_id, start, end in build_trial_slices(len(probs)):
            scores[(subject_id, trial_id)] = float(probs[start:end, 1].mean())
            if labels is not None:
                label_map[(subject_id, trial_id)] = majority_label(labels[start:end])
    return scores, label_map


def combine_scores(score_maps, weight_map, label_map):
    common_keys = None
    for score_map in score_maps.values():
        key_set = set(score_map.keys())
        if common_keys is None:
            common_keys = key_set
        else:
            common_keys = common_keys & key_set

    if not common_keys:
        raise ValueError("No common (user_id, trial_id) pairs were found across the selected models.")

    rows = []
    for subject_id, trial_id in sorted(common_keys, key=subject_trial_sort_key):
        row = {
            "user_id": subject_id,
            "trial_id": int(trial_id),
        }
        weighted_sum = 0.0
        total_weight = 0.0
        for model_name in score_maps:
            prob = float(score_maps[model_name][(subject_id, trial_id)])
            row[model_name] = prob
            weighted_sum += prob * weight_map[model_name]
            total_weight += weight_map[model_name]
        fused_prob = weighted_sum / total_weight
        row["fused_prob_1"] = fused_prob
        row["Emotion_label"] = 1 if fused_prob >= 0.5 else 0
        label_key = (subject_id, trial_id)
        if label_key in label_map:
            row["target_label"] = int(label_map[label_key])
        rows.append(row)
    return rows


def compute_accuracy(rows, column_name):
    labeled_rows = [row for row in rows if "target_label" in row]
    if not labeled_rows:
        return None
    correct = 0
    for row in labeled_rows:
        if column_name == "Emotion_label":
            pred = int(row["Emotion_label"])
        else:
            pred = 1 if row[column_name] >= 0.5 else 0
        if pred == row["target_label"]:
            correct += 1
    return correct / len(labeled_rows)


def write_rows(rows, model_names, out_csv):
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["user_id", "trial_id", "Emotion_label"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def main():
    args = parse_args()
    device_name = args.device
    if args.device != "cpu" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    score_maps = {}
    weight_map = {}
    label_map = {}
    de_subjects = None
    mat_files = None

    if args.edge_ckpt or args.labram_ckpt:
        de_subjects = find_de_subjects(args.de_data_dir, args.extra_data_dir)
        print("DE data sources:", [args.de_data_dir] + list(args.extra_data_dir))
        print("Loaded DE subjects:", len(de_subjects))

    if args.eeg_ckpt:
        mat_files = find_mat_files(args.mat_data_dir, args.extra_data_dir)
        print("EEG data sources:", [args.mat_data_dir] + list(args.extra_data_dir))
        print("Loaded EEG subjects:", len(mat_files))

    if args.edge_ckpt:
        print("Loading EdgeConv...")
        edge_model = load_edge_model(args, device)
        edge_scores, edge_labels = run_edge_model(args, edge_model, device, de_subjects)
        score_maps["edgeconv"] = edge_scores
        weight_map["edgeconv"] = args.edge_weight
        merge_label_maps(label_map, edge_labels, "edgeconv")

    if args.labram_ckpt:
        print("Loading LaBraM...")
        labram_model, input_chans = load_labram_model(args, device)
        labram_scores, labram_labels = run_labram_model(args, labram_model, input_chans, device, de_subjects)
        score_maps["labram"] = labram_scores
        weight_map["labram"] = args.labram_weight
        merge_label_maps(label_map, labram_labels, "labram")

    if args.eeg_ckpt:
        print("Loading EEG-Conformer...")
        eeg_model = load_eeg_model(args, device)
        eeg_scores, eeg_labels = run_eeg_model(args, eeg_model, device, mat_files)
        score_maps["eeg_conformer"] = eeg_scores
        weight_map["eeg_conformer"] = args.eeg_weight
        if not label_map:
            merge_label_maps(label_map, eeg_labels, "eeg_conformer")

    if not score_maps:
        raise ValueError("Please provide at least one checkpoint path.")

    rows = combine_scores(score_maps, weight_map, label_map)
    model_names = list(score_maps.keys())
    write_rows(rows, model_names, args.out_csv)

    print("Saved:", args.out_csv)
    print("Common user-trial pairs:", len(rows))
    for model_name in model_names:
        acc = compute_accuracy(rows, model_name)
        if acc is None:
            print("%s trial accuracy: N/A (labels unavailable)" % model_name)
        else:
            print("%s trial accuracy: %.4f" % (model_name, acc))
    fused_acc = compute_accuracy(rows, "Emotion_label")
    if fused_acc is None:
        print("fused trial accuracy: N/A (labels unavailable)")
    else:
        print("fused trial accuracy: %.4f" % fused_acc)


if __name__ == "__main__":
    main()
