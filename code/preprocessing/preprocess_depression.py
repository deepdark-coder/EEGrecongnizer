"""
Extract DE features from depression .mat files (v7.3 HDF5 format).

Same processing as DE_process.py: 5-band Butterworth filtering → 1s windows → DE → per-subject z-score.
Output: {subject}_X.npy (T, 30, 5) and {subject}_Y.npy (T,)

Usage:
  python preprocess_depression.py
"""
import os
import numpy as np
import scipy.signal as signal
import h5py
import math
from pathlib import Path


def calculate_de(variance):
    clipped = np.clip(variance, 1e-10, None)
    return 0.5 * np.log(2 * math.pi * math.e * clipped)


def extract_de_features(eeg_data, fs=250, window_size_sec=1.0):
    data = np.array(eeg_data, copy=True)
    num_channels, num_points = data.shape
    window_pts = int(fs * window_size_sec)
    num_windows = num_points // window_pts

    bands = {
        'delta': (1, 4), 'theta': (4, 8),
        'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)
    }

    de_features = np.zeros((num_windows, num_channels, 5))

    for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
        b, a = signal.butter(N=4, Wn=[low, high], btype='bandpass', fs=fs)
        filtered = signal.filtfilt(b, a, data, axis=1)

        for w in range(num_windows):
            start = w * window_pts
            end = start + window_pts
            window_data = filtered[:, start:end]
            variance = np.var(window_data, axis=1)
            de_features[w, :, band_idx] = calculate_de(variance)

    return de_features


def process_subject(mat_path, output_dir):
    subject_name = mat_path.stem
    print(f'Processing {subject_name} ...')

    with h5py.File(str(mat_path), 'r') as f:
        # h5py stores MATLAB arrays transposed: (30, 50000) → need .T → (50000, 30)
        eeg_pos = np.array(f['EEG_data_pos']).T  # (50000, 30)
        eeg_neu = np.array(f['EEG_data_neu']).T  # (50000, 30)

    eeg_pos = np.squeeze(eeg_pos)
    eeg_neu = np.squeeze(eeg_neu)

    de_pos = extract_de_features(eeg_pos)
    de_neu = extract_de_features(eeg_neu)

    labels_pos = np.ones(de_pos.shape[0], dtype=np.int64)
    labels_neu = np.zeros(de_neu.shape[0], dtype=np.int64)

    X = np.concatenate([de_pos, de_neu], axis=0).astype(np.float32)
    Y = np.concatenate([labels_pos, labels_neu], axis=0)

    # Per-subject z-score (in-place on float32 copy)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mean) / std

    np.save(output_dir / f'{subject_name}_X.npy', X)
    np.save(output_dir / f'{subject_name}_Y.npy', Y)
    print(f'  Done: X={X.shape}, Y={Y.shape}')


def main():
    input_dir = Path(r'D:\Emotion\4_dataset\4_dataset\traindata\yiyuzheng')
    output_dir = Path(r'D:\Emotion\4_dataset_Processed\4_dataset_Processed\train\depression')
    output_dir.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(input_dir.glob('*.mat'))
    print(f'Found {len(mat_files)} depression .mat files')

    for mat_file in mat_files:
        try:
            process_subject(mat_file, output_dir)
        except Exception as e:
            print(f'  FAILED: {e}')


if __name__ == '__main__':
    main()
