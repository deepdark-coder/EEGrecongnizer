"""
DGCNN with Chebyshev Spectral Graph Convolution for EEG Emotion Recognition.

Architecture: Chebyshev K-order GCN layers → Flatten → FC head
Graph: Learned adjacency matrix A, normalized via symmetric Laplacian
Performance: ~80.11% (stride=1, K=25, 64d)

Reference: TGCN (Zhong et al., 2020) + Chebyshev spectral filtering
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, num_in, num_out, bias=False):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.weight = nn.Parameter(torch.FloatTensor(num_in, num_out))
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(num_out))
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        return out


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class Chebynet(nn.Module):
    """K-order Chebyshev graph convolution layer."""
    def __init__(self, in_features, K, num_out, dropout=0.0):
        super().__init__()
        self.K = K
        self.gc = nn.ModuleList()
        for _ in range(K):
            self.gc.append(GraphConvolution(in_features, num_out))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, L):
        device = x.device
        adj = self._generate_cheby_adj(L, self.K, device)
        result = None
        for i in range(self.K):
            out = self.gc[i](x, adj[i])
            if result is None:
                result = out
            else:
                result += out
        result = self.dropout(result)
        return F.relu(result)

    @staticmethod
    def _generate_cheby_adj(A, K, device):
        support = []
        for i in range(K):
            if i == 0:
                support.append(torch.eye(A.shape[1]).to(device))
            elif i == 1:
                support.append(A)
            else:
                support.append(torch.matmul(support[-1], A))
        return support


class DGCNN(nn.Module):
    """DGCNN for EEG emotion recognition with Chebyshev spectral GCN.

    Args:
        in_features: input features per node (default 5 for DE bands)
        num_nodes: number of EEG channels (default 30)
        k_adj: Chebyshev polynomial order (default 10, best 25 for stride=1)
        num_out: GCN hidden dim (default 64)
        nclass: number of classes (default 2)
        gcn_dropout: dropout after each Chebynet layer
        num_gcn_layers: number of stacked Chebynet layers (default 2)
    """
    def __init__(self, in_features=5, num_nodes=30, k_adj=10, num_out=64, nclass=2,
                 gcn_dropout=0.2, num_gcn_layers=2):
        super().__init__()
        self.K = k_adj
        self.num_out = num_out
        self.num_nodes = num_nodes
        self.num_gcn_layers = num_gcn_layers

        self.bn_input = nn.BatchNorm1d(in_features)

        self.gcn_layers = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()
        for i in range(num_gcn_layers):
            in_dim = in_features if i == 0 else num_out
            self.gcn_layers.append(Chebynet(in_dim, k_adj, num_out, dropout=gcn_dropout))
            self.gcn_bns.append(nn.BatchNorm1d(num_out))

        flat_dim = num_nodes * num_out
        self.fc1 = Linear(flat_dim, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = Linear(64, nclass)

        self.A = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        nn.init.xavier_normal_(self.A)

    def forward(self, x):
        x = self.bn_input(x.transpose(1, 2)).transpose(1, 2)
        L = self._normalize_A(self.A)
        for i in range(self.num_gcn_layers):
            x = self.gcn_layers[i](x, L)
            x = self.gcn_bns[i](x.transpose(1, 2)).transpose(1, 2)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop2(x)
        return self.fc3(x)

    @staticmethod
    def _normalize_A(A):
        A = F.relu(A)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
        return L
