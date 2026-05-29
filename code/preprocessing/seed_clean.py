"""
Clean SEED DE data to match our dataset format (30 channels, 5 bands, binary labels).

Input: SEED DE features (DE/session{1,2,3}/*.mat)
  - Shape: (62, T, 5), labels: -1/0/1 (negative/neutral/positive)

Output: Our format (T, 30, 5) per virtual subject, labels: 0/1
  - Saved as *_X.npy / *_Y.npy in output_dir

Channel mapping: SEED 62ch → our 30ch (standard 10-20 subset)
"""
import os
import sys
import numpy as np
from pathlib import Path
from scipy.stats import zscore


# SEED 62-channel order (standard extended 10-20):
SEED_62 = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
    'O2', 'CB2'
]

# Our 30-channel order:
OUR_30 = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ',
    'FC4', 'FT8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'TP7', 'CP3', 'CPZ',
    'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2'
]

# T3=T7, T4=T8, T5=P7, T6=P8 in standard 10-20
NAME_MAP = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}


def build_channel_index():
    """Build SEED 62-channel → our 30-channel index mapping."""
    seed_idx = {}
    for i, name in enumerate(SEED_62):
        seed_idx[name] = i

    our_indices = []
    for name in OUR_30:
        seed_name = NAME_MAP.get(name, name)
        if seed_name in seed_idx:
            our_indices.append(seed_idx[seed_name])
        else:
            print(f'WARNING: {name} ({seed_name}) not found in SEED 62!')
            our_indices.append(0)

    return np.array(our_indices)


def clean_seed_session(de_dir, output_dir, window_per_subject=400):
    """Process one SEED session directory.

    Each SEED .mat file (1 subject) → multiple virtual subjects (400 windows each).
    """
    ch_idx = build_channel_index()
    print(f'Channel mapping: {len(ch_idx)} channels selected from 62')

    de_path = Path(de_dir)
    mat_files = sorted(de_path.glob('*.mat'))
    mat_files = [f for f in mat_files if f.name != 'label.mat']
    print(f'Found {len(mat_files)} .mat files in {de_dir}')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    virtual_id = 0
    total_windows = 0

    for mat_file in mat_files:
        import scipy.io as sio
        data = sio.loadmat(str(mat_file))

        if 'DE' not in data:
            print(f'  Skipping {mat_file.name}: no DE key')
            continue

        de = data['DE']       # (62, T, 5)
        labels = data['labelAll'].flatten()  # (T,)

        # Select 30 channels and transpose to (T, 30, 5)
        de_30 = de[ch_idx]           # (30, T, 5)
        de_30 = de_30.transpose(1, 0, 2)  # (T, 30, 5)

        # Filter binary labels: keep -1 (negative→0) and 1 (positive→1)
        binary_mask = (labels != 0)
        de_binary = de_30[binary_mask]
        labels_binary = labels[binary_mask]
        labels_binary = (labels_binary == 1).astype(np.int64)  # 1→1, -1→0

        # z-score per "subject-session"
        de_binary = zscore(de_binary, axis=0)

        T = len(de_binary)
        if T < window_per_subject:
            print(f'  Skipping {mat_file.name}: only {T} binary windows (need {window_per_subject})')
            continue

        # Segment into 400-window virtual subjects (overlapping, balanced)
        stride = window_per_subject // 4  # 75% overlap for more data
        n_segments = (T - window_per_subject) // stride + 1

        for i in range(n_segments):
            start = i * stride
            end = start + window_per_subject
            segment_x = de_binary[start:end]
            segment_y = labels_binary[start:end]

            # Require at least 30% of each class for a valid segment
            n_pos = (segment_y == 1).sum()
            n_neg = (segment_y == 0).sum()
            if n_pos < window_per_subject * 0.3 or n_neg < window_per_subject * 0.3:
                continue

            subj_id = f'SEED{mat_file.stem.replace("_", "")}_{virtual_id:03d}'
            np.save(output_path / f'{subj_id}timedata_X.npy', segment_x.astype(np.float32))
            np.save(output_path / f'{subj_id}timedata_Y.npy', segment_y.astype(np.int64))
            virtual_id += 1
            total_windows += window_per_subject

    print(f'Created {virtual_id} virtual subjects ({total_windows} total windows)')
    return virtual_id


def main():
    base = Path(r'D:\Emotion\DGCNN-main\DGCNN-main\DE')
    output_base = Path(r'D:\Emotion\4_dataset_Processed\4_dataset_Processed\train\seed_clean')

    total = 0
    for session in ['session1', 'session2', 'session3']:
        de_dir = base / session
        if de_dir.exists():
            out_dir = output_base / session
            print(f'\n=== Processing {session} ===')
            n = clean_seed_session(str(de_dir), str(out_dir))
            total += n

    print(f'\nTotal virtual subjects: {total}')
    print('Done!')


if __name__ == '__main__':
    main()
