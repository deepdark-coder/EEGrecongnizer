"""Extract raw EEG windows from .mat files for end-to-end deep learning.

Output: (400, 30, 250) per subject — 200 windows per class × 30 channels × 250 time points.
"""
import numpy as np
import h5py
from pathlib import Path


def extract_raw_windows(eeg_data, fs=250, window_size_sec=1.0):
    """Segment raw EEG into non-overlapping windows. (30, T) -> (N, 30, 250)"""
    data = np.array(eeg_data, copy=True)
    num_channels, num_points = data.shape
    window_pts = int(fs * window_size_sec)
    num_windows = num_points // window_pts
    windows = np.zeros((num_windows, num_channels, window_pts), dtype=np.float32)

    for w in range(num_windows):
        start = w * window_pts
        end = start + window_pts
        windows[w] = data[:, start:end]

    return windows


def per_channel_zscore(data):
    """Z-score normalize per channel across all windows."""
    # data: (N, C, T)
    mean = data.mean(axis=(0, 2), keepdims=True)
    std = data.std(axis=(0, 2), keepdims=True) + 1e-6
    return (data - mean) / std


def main():
    import sys
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(input_dir.glob('*.mat')):
        name = f.stem
        print(f'Processing {name} ...', end=' ')
        try:
            with h5py.File(str(f), 'r') as hf:
                pos = np.array(hf['EEG_data_pos']).T
                neu = np.array(hf['EEG_data_neu']).T

            pos_win = extract_raw_windows(np.squeeze(pos))
            neu_win = extract_raw_windows(np.squeeze(neu))

            X = np.concatenate([pos_win, neu_win], axis=0)  # (400, 30, 250)
            Y = np.concatenate([np.ones(len(pos_win), dtype=int),
                                np.zeros(len(neu_win), dtype=int)])

            X = per_channel_zscore(X)

            np.save(output_dir / f'{name}_X.npy', X)
            np.save(output_dir / f'{name}_Y.npy', Y)
            print(f'OK: {X.shape}')
        except Exception as e:
            print(f'FAIL: {e}')


if __name__ == '__main__':
    main()
