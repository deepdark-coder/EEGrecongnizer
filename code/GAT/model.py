"""
Graph Attention Network (GAT) DGCNN for EEG Emotion Recognition.

Architecture: 2 stacked GAT layers (k-NN + learned attention weights) → Flatten → FC head
vs EdgeConv: replaces max-pool aggregation with soft attention-weighted sum over neighbors.

Performance:
  GAT baseline:                   80.35% +- 4.93%
  GAT + Depression extra:         81.79% +- 4.93%  (+1.44%)

Key lesson: max-pool > attention for noisy EEG. Hard selection is more robust
than soft attention which gets distracted by noise.
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


class GATLayer(nn.Module):
    """Graph Attention layer: k-NN neighbor aggregation with learned attention weights.

    Same k-NN + edge feature construction as EdgeConvLayer, but replaces
    max-pool with attention-weighted sum over neighbors.
    """
    def __init__(self, in_features, out_features, k=20, spatial_dist=None):
        super().__init__()
        self.k = k
        self.value_conv = nn.Conv2d(2 * in_features, out_features, kernel_size=1, bias=False)
        self.att_conv = nn.Conv2d(2 * in_features, 1, kernel_size=1, bias=False)
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

        value = self.value_conv(edge_feat)
        att = self.att_conv(edge_feat)
        att = self.leaky_relu(att)
        att = F.softmax(att, dim=-1)

        out = (value * att).sum(dim=-1)
        out = out.permute(0, 2, 1)
        return out


class GATDGCNN(nn.Module):
    """DGCNN with GAT aggregation for EEG emotion recognition.

    Args:
        in_features: input features per node
        num_nodes: number of EEG channels (default 30)
        k: k nearest neighbors (default 20)
        nclass: number of classes (default 2)
        spatial_dist: (N, N) spatial distance matrix or None
        return_features: if True, forward returns (logits, features) for DANN
    """
    def __init__(self, in_features=5, num_nodes=30, k=20, nclass=2,
                 spatial_dist=None, return_features=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.k = k
        self.return_features = return_features

        self.bn_input = nn.BatchNorm1d(in_features)

        self.gat1 = GATLayer(in_features, 64, k, spatial_dist)
        self.bn1 = nn.BatchNorm1d(64)
        self.gat2 = GATLayer(64, 64, k, spatial_dist)
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
        x = self.bn_input(x.transpose(1, 2)).transpose(1, 2)

        x = self.gat1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.gat2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)

        features = x.reshape(x.shape[0], -1)
        x = F.relu(self.bn_fc1(self.fc1(features)))
        x = self.drop1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop2(x)
        logits = self.fc3(x)

        if self.return_features:
            return logits, features
        return logits


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class SubjectClassifier(nn.Module):
    def __init__(self, in_features, num_subjects):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, num_subjects),
        )

    def forward(self, x):
        return self.fc(x)


class GATDGCNN_DANN(nn.Module):
    """GATDGCNN with adversarial subject adaptation."""
    def __init__(self, in_features=5, num_nodes=30, k=20, nclass=2,
                 num_subjects=40, spatial_dist=None):
        super().__init__()
        self.encoder = GATDGCNN(in_features, num_nodes, k, nclass,
                                spatial_dist, return_features=True)
        self.subject_cls = SubjectClassifier(num_nodes * 64, num_subjects)

    def forward(self, x, lambda_=0.0):
        logits, features = self.encoder(x)
        if lambda_ > 0:
            reversed_features = GradientReversal.apply(features, lambda_)
            subj_logits = self.subject_cls(reversed_features)
            return logits, subj_logits
        return logits
