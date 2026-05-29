"""
EdgeConv DGCNN for EEG Emotion Recognition — Our SOTA (82.76% / 83.92% w/ Depression).

Architecture: 2 stacked dynamic k-NN EdgeConv layers → Flatten → FC head
Key innovation: dynamic graph construction per sample (learns functional connectivity)
vs Chebyshev GCN's fixed learned adjacency.

Variants: Band SE, Spatial Prior, DANN (adversarial subject adaptation), SupCon (contrastive)

Performance:
  EdgeConv v1 stride=1:              82.76% +- 4.96%  (original SOTA)
  EdgeConv v1 + Depression extra:     83.92% +- 4.39%  (new SOTA, +1.16%)
  EdgeConv v1 3-seed avg:            82.23%            (most robust)
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
    """Dynamic edge convolution layer with optional spatial prior.

    For each node, finds k nearest neighbors in a fused distance space:
        fused_dist = feature_dist + beta * spatial_dist
    where beta is a learnable scalar controlling spatial prior strength.

    Then applies Conv2d+BN+LeakyReLU to edge features [x_j - x_i, x_i].
    """
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


class BandSEBlock(nn.Module):
    """Squeeze-and-Excitation over EEG frequency bands.

    Learns per-band importance: delta/theta/alpha/beta/gamma contribute
    differently to emotion (alpha asymmetry is the strongest valence marker).
    """
    def __init__(self, n_bands=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_bands, n_bands),
            nn.ReLU(),
            nn.Linear(n_bands, n_bands),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, N, Bands, T = x.shape
        w = x.mean(dim=[1, 3])
        w = self.fc(w)
        return x * w.view(B, 1, Bands, 1)


class EdgeDGCNN(nn.Module):
    """EdgeConv DGCNN for EEG emotion recognition.

    Args:
        in_features: input features per node
        num_nodes: number of EEG channels (default 30)
        k: k nearest neighbors for EdgeConv (default 20)
        nclass: number of classes (default 2)
        spatial_dist: (N, N) spatial distance matrix or None
        return_features: if True, forward returns (logits, features) for DANN
        use_supcon: if True, forward returns (logits, normalized_projection)
        use_band_se: enable band-level SE attention
        n_bands: number of frequency bands (default 5)
    """
    def __init__(self, in_features=5, num_nodes=30, k=20, nclass=2,
                 spatial_dist=None, return_features=False, use_supcon=False,
                 use_band_se=False, n_bands=5):
        super().__init__()
        self.num_nodes = num_nodes
        self.k = k
        self.return_features = return_features
        self.use_supcon = use_supcon
        self.use_band_se = use_band_se
        self.n_bands = n_bands

        self.bn_input = nn.BatchNorm1d(in_features)

        if use_band_se:
            self.band_se = BandSEBlock(n_bands)

        self.edgeconv1 = EdgeConvLayer(in_features, 64, k, spatial_dist)
        self.bn1 = nn.BatchNorm1d(64)
        self.edgeconv2 = EdgeConvLayer(64, 64, k, spatial_dist)
        self.bn2 = nn.BatchNorm1d(64)

        flat_dim = num_nodes * 64
        if use_supcon:
            self.projection = nn.Sequential(
                nn.Linear(flat_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
        self.fc1 = Linear(flat_dim, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = Linear(64, nclass)

    def forward(self, x):
        x = self.bn_input(x.transpose(1, 2)).transpose(1, 2)

        if self.use_band_se:
            B, N, C = x.shape
            T = C // self.n_bands
            x = x.reshape(B, N, self.n_bands, T)
            x = self.band_se(x)
            x = x.reshape(B, N, C)

        x = self.edgeconv1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.edgeconv2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)

        features = x.reshape(x.shape[0], -1)
        x = F.relu(self.bn_fc1(self.fc1(features)))
        x = self.drop1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop2(x)
        logits = self.fc3(x)

        if self.use_supcon:
            proj = F.normalize(self.projection(features), dim=1)
            return logits, proj
        if self.return_features:
            return logits, features
        return logits


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for adversarial domain adaptation."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class SubjectClassifier(nn.Module):
    """MLP that classifies subject ID from features (for adversarial training)."""
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


class EdgeDGCNN_DANN(nn.Module):
    """EdgeDGCNN with adversarial subject adaptation."""
    def __init__(self, in_features=5, num_nodes=30, k=20, nclass=2,
                 num_subjects=40, spatial_dist=None):
        super().__init__()
        self.encoder = EdgeDGCNN(in_features, num_nodes, k, nclass,
                                 spatial_dist, return_features=True)
        self.subject_cls = SubjectClassifier(num_nodes * 64, num_subjects)

    def forward(self, x, lambda_=0.0):
        logits, features = self.encoder(x)
        if lambda_ > 0:
            reversed_features = GradientReversal.apply(features, lambda_)
            subj_logits = self.subject_cls(reversed_features)
            return logits, subj_logits
        return logits


@torch.no_grad()
def adapt_bn(model, loader, device):
    """AdaBN: replace BN running stats with target domain (test subject) statistics."""
    all_x = []
    for batch in loader:
        all_x.append(batch[0])
    all_x = torch.cat(all_x, dim=0).float().to(device)

    bn_modules = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_modules.append(m)
            m.momentum = 1.0
            m.train()

    model(all_x)

    for m in bn_modules:
        m.momentum = 0.1
        m.eval()
