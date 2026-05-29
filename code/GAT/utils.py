import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class eegDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, s_tensor=None):
        self.x = x_tensor
        self.y = y_tensor
        self.s = s_tensor

    def __getitem__(self, index):
        if self.s is not None:
            return self.x[index], self.y[index], self.s[index]
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def load_subject_data(data_dir=None):
    """Load all subject DE data from npy files.

    Returns:
        subjects: list of (X, Y) tuples, one per subject
        subject_ids: list of subject ID strings
    """
    if data_dir is not None:
        dirs = [data_dir]
    else:
        dirs = _find_data_dirs()

    subjects = []
    subject_ids = []

    for d in dirs:
        data_path = Path(d)
        x_files = sorted(data_path.glob('*_X.npy'))
        for x_file in x_files:
            y_file = data_path / x_file.name.replace('_X.npy', '_Y.npy')
            if not y_file.exists():
                continue
            x_data = np.load(x_file)
            y_data = np.load(y_file)
            subjects.append((x_data, y_data))
            subject_ids.append(x_file.stem.replace('timedata_X', ''))

    return subjects, subject_ids


def build_windows(subject_data, subject_labels, window_size=1, stride=1):
    """Build samples with optional temporal window splicing.

    Args:
        subject_data: (T, 30, 5) DE features for one subject
        subject_labels: (T,) labels
        window_size: number of consecutive windows to splice
        stride: sliding step

    Returns:
        X: (N, 30, 5 * window_size) or (N, 30, 5) if window_size=1
        Y: (N,) labels (majority vote per window)
    """
    T = len(subject_data)
    if window_size == 1:
        return subject_data, subject_labels

    samples_x = []
    samples_y = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        window = subject_data[start:end]
        window = window.transpose(1, 0, 2).reshape(30, -1)
        label = int(np.bincount(subject_labels[start:end]).argmax())
        samples_x.append(window)
        samples_y.append(label)

    return np.array(samples_x), np.array(samples_y)


# 10-20 symmetric electrode pair indices (left, right) for the 30-channel layout
HEMISPHERIC_PAIRS = [
    (0, 1), (2, 6), (3, 5), (7, 11), (8, 10),
    (12, 16), (13, 15), (17, 21), (18, 20),
    (22, 26), (23, 25), (27, 29),
]


def add_asymmetry_channels(x):
    """Append left-right differential channels for hemispheric asymmetry.

    Args:
        x: (N, 30, F) numpy array
    Returns:
        (N, 42, F) numpy array — 30 original + 12 asymmetry channels
    """
    left_idx = [l for l, r in HEMISPHERIC_PAIRS]
    right_idx = [r for l, r in HEMISPHERIC_PAIRS]
    asym = x[:, left_idx, :] - x[:, right_idx, :]
    return np.concatenate([x, asym], axis=1)


def _find_data_dirs():
    candidates = [
        r'D:\Emotion\4_dataset_Processed',
        '/home/kxy/Tanhuafu/data',
    ]
    dirs = set()
    for base in candidates:
        base_path = Path(base)
        if base_path.exists():
            for match in base_path.rglob('*_X.npy'):
                dirs.add(str(match.parent))
    if not dirs:
        raise FileNotFoundError(f'No *_X.npy files found under candidates')
    return sorted(dirs)
