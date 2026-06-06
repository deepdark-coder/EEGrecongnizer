import os
import torch
import numpy as np
from torch.utils.data import Dataset


class CompetitionDEDataset(Dataset):
    """
    DE feature dataset for LaBraM (TemporalConv path, in_chans=1).
    Groups N consecutive DE windows (each 30 ch x 5 bands) into (30, N*5) time series.
    Default N=40 gives (30, 200) — compatible with LaBraM's input_size=200.
    """
    def __init__(self, data_dir, mode='train', files=None, window_size=40, stride=10):
        super().__init__()
        self.mode = mode
        self.data = []
        self.labels = []

        self.ch_names = [
            'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ',
            'FC4', 'FT8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'TP7', 'CP3', 'CPZ',
            'CP4', 'TP8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2'
        ]

        print(f"Loading {mode} DE dataset (window_size={window_size}, stride={stride})...")
        self._load_data(data_dir, files, window_size, stride)

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
        print(f"{self.mode} DE dataset built: {len(self.labels)} samples, shape {self.data.shape}")

    def _load_data(self, data_dir, files, window_size, stride):
        if files is not None:
            file_list = sorted(files)
        else:
            all_files = [f for f in os.listdir(data_dir) if f.endswith('_X.npy')]
            file_list = sorted(all_files)

        if len(file_list) == 0:
            raise FileNotFoundError(f"No DE .npy files found in {data_dir}!")

        for file in file_list:
            base = file.replace('_X.npy', '')
            x_path = os.path.join(data_dir, f'{base}_X.npy')
            y_path = os.path.join(data_dir, f'{base}_Y.npy')

            x = np.load(x_path)  # (400, 30, 5)
            y = np.load(y_path)  # (400,)

            for label in [0, 1]:
                idx = np.where(y == label)[0]
                x_class = x[idx]  # (200, 30, 5)

                for start in range(0, len(x_class) - window_size + 1, stride):
                    chunk = x_class[start:start + window_size]  # (window_size, 30, 5)
                    sample = chunk.transpose(1, 0, 2).reshape(30, -1)
                    self.data.append(sample)
                    self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
