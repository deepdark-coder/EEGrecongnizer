"""
Mixture of Graph Experts (MoGE) with EdgeConv for EEG Emotion Recognition.

Architecture: Per-channel router → K parallel EdgeConv experts → gated sum → shared EdgeConv → FC head
Key idea: Different EEG channels (frontal, temporal, occipital) learn different interaction
patterns via expert specialization.

Performance: MoGE_EdgeDGCNN: 81.05% (ties with EdgeConv v1 stride=3 baseline)

Reference: Xuanhao Liu et al., BIBM 2024
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
    """Dynamic edge convolution layer with optional spatial prior."""
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


class MoGE_EdgeDGCNN(nn.Module):
    """Mixture of Graph Experts with EdgeConv backbone.

    Per-channel router assigns each EEG channel to an EdgeConv expert via soft gating.
    Different experts learn different channel interaction patterns.

    Args:
        in_features: input features per node
        num_nodes: number of EEG channels (default 30)
        k: k nearest neighbors for EdgeConv (default 20)
        hidden_dim: output dim per expert (default 64)
        num_experts: number of graph experts (default 3)
        nclass: number of classes (default 2)
        spatial_dist: optional (N, N) spatial distance matrix
    """
    def __init__(self, in_features=5, num_nodes=30, k=20, hidden_dim=64,
                 num_experts=3, nclass=2, spatial_dist=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_experts = num_experts

        self.bn_input = nn.BatchNorm1d(in_features)

        self.router = nn.Linear(in_features, num_experts)

        self.experts = nn.ModuleList([
            EdgeConvLayer(in_features, hidden_dim, k, spatial_dist)
            for _ in range(num_experts)
        ])
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.edgeconv2 = EdgeConvLayer(hidden_dim, hidden_dim, k, spatial_dist)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        flat_dim = num_nodes * hidden_dim
        self.fc1 = Linear(flat_dim, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = Linear(64, nclass)

    def forward(self, x):
        x = self.bn_input(x.transpose(1, 2)).transpose(1, 2)

        gate = F.softmax(self.router(x), dim=-1)  # (B, N, E)

        out = None
        for e in range(self.num_experts):
            x_e = self.experts[e](x)
            g = gate[:, :, e:e+1]
            if out is None:
                out = x_e * g
            else:
                out += x_e * g

        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        out = self.edgeconv2(out)
        out = self.bn2(out.transpose(1, 2)).transpose(1, 2)

        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.bn_fc1(self.fc1(out)))
        out = self.drop1(out)
        out = F.relu(self.bn_fc2(self.fc2(out)))
        out = self.drop2(out)
        return self.fc3(out)
