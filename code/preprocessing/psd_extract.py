"""
Extract high-resolution PSD features from raw EEG .mat files.

Replaces 5-band DE with 25 narrow bands (2Hz each, 1-50Hz).
Preserves more spectral detail while keeping the same time-window structure.

Output: (T, 30, 25) per subject — same format as DE but 25 bands instead of 5.
"""
import numpy as np
import scipy.signal as signal
import h5py
from pathlib import Path


def extract_psd_features(eeg_data, fs=250, window_size_sec=1.0, n_bands=25):
    """
    Extract PSD features with fine frequency resolution.

    Args:
        eeg_data: (30, num_points) raw EEG
        fs: sampling rate (250 Hz)
        window_size_sec: sliding window in seconds
        n_bands: number of frequency bands

    Returns:
        psd_features: (num_windows, 30, n_bands)
    """
    data_safe = np.array(eeg_data, copy=True)
    num_channels, num_points = data_safe.shape
    window_pts = int(fs * window_size_sec)
    num_windows = num_points // window_pts

    # Define narrow frequency bands: 2Hz steps from 1 to 50Hz
    # 1-3, 3-5, 5-7, ..., 47-49, 49-51 → but last band is 49-50 capped
    band_edges = np.linspace(1, 50, n_bands + 1)
    bands = [(band_edges[i], band_edges[i + 1]) for i in range(n_bands)]

    psd_features = np.zeros((num_windows, num_channels, n_bands))

    for band_idx, (low, high) in enumerate(bands):
        # Bandpass filter
        b, a = signal.butter(N=4, Wn=[low, high], btype='bandpass', fs=fs)
        filtered = signal.filtfilt(b, a, data_safe, axis=1)

        for w in range(num_windows):
            start = w * window_pts
            end = start + window_pts
            window_data = filtered[:, start:end]

            # Compute power (variance of filtered signal)
            power = np.var(window_data, axis=1)

            # Log-transform (like DE)
            psd_features[w, :, band_idx] = np.log(power + 1e-10)

    return psd_features


def batch_process(input_dir_str, output_dir_str):
    """Process all .mat files in input_dir, saving to output_dir."""
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(input_dir.glob('*.mat'))
    print(f'Found {len(mat_files)} .mat files in {input_dir}')

    for file_path in mat_files:
        subject_name = file_path.stem
        print(f'Processing {subject_name} ...', end=' ')

        try:
            with h5py.File(str(file_path), 'r') as f:
                eeg_pos = np.array(f['EEG_data_pos']).T  # (30, 50000)
                eeg_neu = np.array(f['EEG_data_neu']).T

            eeg_pos = np.squeeze(eeg_pos)
            eeg_neu = np.squeeze(eeg_neu)

            psd_pos = extract_psd_features(eeg_pos, fs=250)
            psd_neu = extract_psd_features(eeg_neu, fs=250)

            labels_pos = np.ones(psd_pos.shape[0], dtype=int)
            labels_neu = np.zeros(psd_neu.shape[0], dtype=int)

            X = np.concatenate((psd_pos, psd_neu), axis=0)
            Y = np.concatenate((labels_pos, labels_neu), axis=0)

            # Per-subject z-score
            X_mean = np.mean(X, axis=0, keepdims=True)
            X_std = np.std(X, axis=0, keepdims=True) + 1e-6
            X = (X - X_mean) / X_std

            np.save(output_dir / f'{subject_name}_X.npy', X.astype(np.float32))
            np.save(output_dir / f'{subject_name}_Y.npy', Y.astype(np.int64))

            print(f'OK: {X.shape}')
        except Exception as e:
            print(f'FAILED: {e}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: python psd_extract.py <input_dir> <output_dir>')
        sys.exit(1)
    batch_process(sys.argv[1], sys.argv[2])
