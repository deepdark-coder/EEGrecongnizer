"""
End-to-End Raw Waveform EdgeConv for EEG Emotion Recognition.

Architecture: TemporalCNN (learned per-channel features) → 2× EdgeConv spatial → FC head
Key idea: Replace hand-crafted DE frequency features with learned temporal convolutions
directly on raw 250Hz EEG windows.

Performance: 67.85% — small CNN cannot learn frequency decomposition from limited data.
DE features remain the strong baseline for small-sample EEG.

Lesson: Hand-crafted Butterworth 5-band filtering is a strong inductive bias that
end-to-end learning cannot replicate with only 40 subjects.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class EdgeConvLayer(nn.Module):
    """Dynamic edge convolution layer."""
    def __init__(self, in_features, out_features, k=20, spatial_dist=None):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(2 * in_features, out_features, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        if spatial_dist is not None:
            self.register_buffer('spatial_dist', spatial_dist)
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            self.spatial_dist = None
            self.beta = None

    def forward(self, x):
        B, N, C = x.shape
        k = min(self.k, N)
        dist = torch.cdist(x, x)
        if self.spatial_dist is not None:
            dist = dist + self.beta * self.spatial_dist
        idx = dist.topk(k, largest=False)[1]
        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
        idx_flat = (idx + idx_base).reshape(-1)
        x_flat = x.reshape(-1, C)
        x_j = x_flat[idx_flat].reshape(B, N, k, C)
        x_i = x.unsqueeze(2).expand(-1, -1, k, -1)
        edge_feat = torch.cat([x_j - x_i, x_i], dim=-1)
        edge_feat = edge_feat.permute(0, 3, 1, 2)
        out = self.conv(edge_feat)
        out = self.bn(out)
        out = self.leaky_relu(out)
        out = out.max(dim=-1)[0]
        out = out.permute(0, 2, 1)
        return out


class TemporalCNN(nn.Module):
    """1D CNN that encodes raw EEG time window into feature vector per channel."""
    def __init__(self, in_len=250, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, out_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RawEdgeDGCNN(nn.Module):
    """End-to-end raw EEG model: TemporalCNN per channel -> EdgeConv spatial -> FC.

    Args:
        in_len: raw waveform length (default 250 for 1s at 250Hz)
        num_nodes: number of EEG channels (default 30)
        k: k nearest neighbors (default 20)
        nclass: number of classes (default 2)
        spatial_dist: optional (N, N) spatial distance matrix
    """
    def __init__(self, in_len=250, num_nodes=30, k=20, nclass=2, spatial_dist=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.temporal = TemporalCNN(in_len, 64)

        self.edgeconv1 = EdgeConvLayer(64, 64, k, spatial_dist)
        self.bn1 = nn.BatchNorm1d(64)
        self.edgeconv2 = EdgeConvLayer(64, 64, k, spatial_dist)
        self.bn2 = nn.BatchNorm1d(64)

        flat_dim = num_nodes * 64
        self.fc1 = Linear(flat_dim, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = Linear(64, nclass)

    def forward(self, x):
        B, N, T = x.shape
        x = x.reshape(B * N, 1, T)
        x = self.temporal(x)
        x = x.reshape(B, N, 64)

        x = self.edgeconv1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.edgeconv2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)

        x = x.reshape(B, -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop2(x)
        return self.fc3(x)
