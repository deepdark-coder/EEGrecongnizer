"""Spatial prior: 10-20 EEG channel 3D coordinates for graph regularization.

The 30 channels match the order in all .npy data files: (T, 30, 5).
Coordinates are on a unit sphere (radius~1), based on standard 10-20 system.
T3=T7, T4=T8, T5=P7, T6=P8.
"""
import torch
import numpy as np

CHANNELS_30 = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'FT7', 'FC3', 'FCZ', 'FC4', 'FT8',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'TP7', 'CP3', 'CPZ', 'CP4', 'TP8',
    'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'OZ', 'O2',
]

COORDS_3D = {
    'FP1': (-0.27, 0.85, 0.35), 'FP2': (0.27, 0.85, 0.35),
    'F7':  (-0.67, 0.41, -0.45), 'F3': (-0.41, 0.67, 0.33),
    'FZ':  (0.0, 0.72, 0.31),   'F4': (0.41, 0.67, 0.33),
    'F8':  (0.67, 0.41, -0.45),
    'FT7': (-0.72, 0.15, -0.55), 'FC3': (-0.41, 0.41, 0.55),
    'FCZ': (0.0, 0.45, 0.57),   'FC4': (0.41, 0.41, 0.55),
    'FT8': (0.72, 0.15, -0.55),
    'T7':  (-0.85, 0.0, -0.33), 'C3': (-0.54, 0.0, 0.67),
    'CZ':  (0.0, 0.0, 0.72),    'C4': (0.54, 0.0, 0.67),
    'T8':  (0.85, 0.0, -0.33),
    'TP7': (-0.72, -0.15, -0.55), 'CP3': (-0.41, -0.41, 0.55),
    'CPZ': (0.0, -0.45, 0.57),   'CP4': (0.41, -0.41, 0.55),
    'TP8': (0.72, -0.15, -0.55),
    'P7':  (-0.67, -0.41, -0.45), 'P3': (-0.41, -0.67, 0.33),
    'PZ':  (0.0, -0.72, 0.31),   'P4': (0.41, -0.67, 0.33),
    'P8':  (0.67, -0.41, -0.45),
    'O1':  (-0.27, -0.85, 0.35), 'OZ': (0.0, -0.90, 0.31),
    'O2':  (0.27, -0.85, 0.35),
}


def get_spatial_dist(normalize=True, device=None):
    """Build (30, 30) spatial distance matrix from 10-20 coordinates."""
    coords = np.array([COORDS_3D[ch] for ch in CHANNELS_30], dtype=np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    dist = torch.tensor(dist, dtype=torch.float32, device=device)
    if normalize:
        dist = dist / dist.max()
    return dist
