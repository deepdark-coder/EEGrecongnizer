import argparse
import datetime
import glob
import os

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from sklearn.model_selection import KFold
from torch import Tensor
from torch.backends import cudnn


cudnn.benchmark = False
cudnn.deterministic = True

gpus = [0]


def denoise_signals(all_data):
    std_val = np.std(all_data)
    threshold = 3 * std_val
    all_data = np.clip(all_data, -threshold, threshold)
    common_mode_noise = np.mean(all_data, axis=1, keepdims=True)
    return all_data - common_mode_noise


def pick_group_count(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class SampleChannelNorm(nn.Module):
    def __init__(self, n_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, 1, n_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, 1, n_channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class TemporalBranch(nn.Sequential):
    def __init__(self, out_channels: int, kernel_size: int, in_channels: int = 1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                (1, kernel_size),
                stride=(1, 1),
                padding=(0, kernel_size // 2),
                bias=False,
            ),
            nn.GroupNorm(pick_group_count(out_channels), out_channels),
            nn.ELU(),
        )


class SqueezeExcite2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class AdaptiveSpatialAggregator(nn.Module):
    def __init__(self, in_channels: int, emb_size: int, spatial_heads: int = 2):
        super().__init__()
        hidden_channels = max(16, in_channels // 2)
        self.attn_logits = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, (1, 1), bias=False),
            nn.GroupNorm(pick_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, spatial_heads, (1, 1), bias=True),
        )
        self.value_proj = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, (1, 1), bias=False),
            nn.GroupNorm(pick_group_count(emb_size), emb_size),
            nn.SiLU(),
        )
        self.token_dropout = nn.Dropout(0.1)
        self.token_norm = nn.LayerNorm(emb_size)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.attn_logits(x)
        attn = torch.softmax(logits / (x.shape[1] ** 0.5), dim=2)
        values = self.value_proj(x)
        pooled = torch.einsum("bhct,bect->beht", attn, values)
        mean_head = values.mean(dim=2, keepdim=True)
        max_head = values.amax(dim=2, keepdim=True)
        tokens = torch.cat([mean_head, max_head, pooled], dim=2)
        tokens = rearrange(tokens, "b e h t -> b (h t) e")
        return self.token_norm(self.token_dropout(tokens))


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40, n_channels: int = 30):
        super().__init__()
        branch_channels = 20
        temporal_channels = branch_channels * 3
        spectral_channels = branch_channels * 4
        feat_channels = 48
        feat_groups = pick_group_count(feat_channels)
        self.input_norm = SampleChannelNorm(n_channels)
        self.temporal_branches = nn.ModuleList(
            [
                TemporalBranch(branch_channels, kernel_size=7),
                TemporalBranch(branch_channels, kernel_size=15),
                TemporalBranch(branch_channels, kernel_size=31),
            ]
        )
        self.temporal_fuse = nn.Sequential(
            nn.Conv2d(temporal_channels, feat_channels, (1, 1), stride=(1, 1), bias=False),
            nn.GroupNorm(feat_groups, feat_channels),
            nn.ELU(),
        )
        self.temporal_context = nn.Sequential(
            nn.Conv2d(
                feat_channels,
                feat_channels,
                (1, 9),
                stride=(1, 1),
                padding=(0, 4),
                groups=feat_channels,
                bias=False,
            ),
            nn.GroupNorm(feat_groups, feat_channels),
            nn.SiLU(),
        )
        self.channel_reweight = SqueezeExcite2d(feat_channels)
        self.temporal_pool = nn.Sequential(
            nn.AvgPool2d((1, 15), stride=(1, 10)),
            nn.Dropout(0.25),
        )
        self.spatial_aggregator = AdaptiveSpatialAggregator(feat_channels, emb_size, spatial_heads=1)

        self.spectral_pointwise = nn.Sequential(
            nn.Conv2d(1, branch_channels, (1, 1), stride=(1, 1), bias=False),
            nn.GroupNorm(pick_group_count(branch_channels), branch_channels),
            nn.SiLU(),
        )
        self.spectral_branches = nn.ModuleList(
            [
                TemporalBranch(branch_channels, kernel_size=3, in_channels=branch_channels),
                TemporalBranch(branch_channels, kernel_size=7, in_channels=branch_channels),
                TemporalBranch(branch_channels, kernel_size=15, in_channels=branch_channels),
            ]
        )
        self.spectral_fuse = nn.Sequential(
            nn.Conv2d(spectral_channels, feat_channels, (1, 1), stride=(1, 1), bias=False),
            nn.GroupNorm(feat_groups, feat_channels),
            nn.ELU(),
        )
        self.spectral_context = nn.Sequential(
            nn.Conv2d(
                feat_channels,
                feat_channels,
                (1, 5),
                stride=(1, 1),
                padding=(0, 2),
                groups=feat_channels,
                bias=False,
            ),
            nn.GroupNorm(feat_groups, feat_channels),
            nn.SiLU(),
        )
        self.spectral_reweight = SqueezeExcite2d(feat_channels)
        self.spectral_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((n_channels, 24)),
            nn.Dropout(0.25),
        )
        self.spectral_aggregator = AdaptiveSpatialAggregator(feat_channels, emb_size, spatial_heads=1)
        self.time_token_type = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)
        self.freq_token_type = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_norm(x)
        time_feat = torch.cat([branch(x) for branch in self.temporal_branches], dim=1)
        time_feat = self.temporal_fuse(time_feat)
        time_feat = self.temporal_context(time_feat)
        time_feat = self.channel_reweight(time_feat)
        time_feat = self.temporal_pool(time_feat)
        time_tokens = self.spatial_aggregator(time_feat)
        time_tokens = time_tokens + self.time_token_type

        spectrum = torch.log1p(torch.abs(torch.fft.rfft(x.float(), dim=-1)))[..., 1:]
        spectral_base = self.spectral_pointwise(spectrum)
        spectral_feat = torch.cat(
            [spectral_base] + [branch(spectral_base) for branch in self.spectral_branches],
            dim=1,
        )
        spectral_feat = self.spectral_fuse(spectral_feat)
        spectral_feat = self.spectral_context(spectral_feat)
        spectral_feat = self.spectral_reweight(spectral_feat)
        spectral_feat = self.spectral_pool(spectral_feat)
        freq_tokens = self.spectral_aggregator(spectral_feat).to(time_tokens.dtype)
        freq_tokens = freq_tokens + self.freq_token_type

        return torch.cat([time_tokens, freq_tokens], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", q, k)
        if mask is not None:
            energy = energy.masked_fill_(~mask, torch.finfo(torch.float32).min)
        att = self.att_drop(F.softmax(energy / (self.emb_size / self.num_heads) ** 0.5, dim=-1))
        out = torch.einsum("bhal, bhlv -> bhav", att, v)
        return self.projection(rearrange(out, "b h n d -> b n (h d)"))


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class EMA:
    def __init__(self, model, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].copy_(self.decay * self.shadow[name] + (1 - self.decay) * param.data)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TokenMixer(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_heads: int = 4,
        drop_p: float = 0.1,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.attn = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
                LayerScale(emb_size),
                DropPath(drop_path),
            )
        )
        self.ffn = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
                LayerScale(emb_size),
                DropPath(drop_path),
            )
        )
        self.final_norm = nn.LayerNorm(emb_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_norm(self.ffn(self.attn(x)))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int, drop_path_max: float = 0.15):
        drop_path_rates = [drop_path_max * i / (depth - 1) for i in range(depth)] if depth > 1 else [0.0]
        super().__init__(*[TokenMixer(emb_size, drop_path=drop_path_rates[i]) for i in range(depth)])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()
        hidden_size = max(64, emb_size)
        self.cls_norm = nn.LayerNorm(emb_size)
        self.token_attention = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(emb_size * 3, hidden_size),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: Tensor):
        cls_feat = self.cls_norm(x[:, 0])
        patch_tokens = x[:, 1:]
        token_scores = self.token_attention(patch_tokens)
        token_weights = torch.softmax(token_scores, dim=1)
        attn_feat = (token_weights * patch_tokens).sum(dim=1)
        mean_feat = patch_tokens.mean(dim=1)
        feat = torch.cat([cls_feat, attn_feat, mean_feat], dim=-1)
        return feat, self.fc(feat)


class ConvPositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            emb_size,
            emb_size,
            kernel_size,
            padding=kernel_size // 2,
            groups=emb_size,
            bias=False,
        )
        self.norm = nn.GroupNorm(pick_group_count(emb_size), emb_size)

    def forward(self, x: Tensor) -> Tensor:
        cls_token = x[:, :1]
        patch_tokens = x[:, 1:]
        pos = self.conv(patch_tokens.transpose(1, 2))
        pos = self.norm(pos).transpose(1, 2)
        return torch.cat([cls_token, patch_tokens + pos], dim=1)


class ViT(nn.Module):
    def __init__(self, emb_size: int, depth: int, n_classes: int = 2, n_channels: int = 30, seq_len: int = 11):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)
        self.pos_encoder = ConvPositionalEncoding(emb_size)
        self.pos_drop = nn.Dropout(0.1)
        self.transformer = TransformerEncoder(depth, emb_size)
        self.cls_head = ClassificationHead(emb_size, n_classes)

    def forward(self, x: Tensor):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(self.pos_encoder(x))
        x = self.transformer(x)
        return self.cls_head(x)


def warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ExGAN:
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        depth: int,
        emb_size: int,
        device: torch.device,
        lr: float = 2e-4,
        weight_decay: float = 5e-4,
        model_seed: int = 42,
        use_denoise: bool = True,
        amp_enabled: bool = True,
        aug_noise_std: float = 0.02,
        aug_shift: int = 25,
        channel_mask_prob: float = 0.3,
        channel_drop_prob: float = 0.1,
        mixup_prob: float = 0.5,
        mixup_alpha: float = 0.1,
        label_smoothing: float = 0.05,
        tta_shift: int = 8,
    ):
        self.n_channels = 30
        self.n_times = 250
        self.n_classes = 2
        self.lr = lr
        self.b1, self.b2 = 0.9, 0.999
        self.weight_decay = weight_decay
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.depth = depth
        self.emb_size = emb_size
        self.device = device
        self.model_seed = model_seed
        self.use_denoise = use_denoise
        self.amp_enabled = amp_enabled and device.type == "cuda"
        self.aug_noise_std = aug_noise_std
        self.aug_shift = aug_shift
        self.channel_mask_prob = channel_mask_prob
        self.channel_drop_prob = channel_drop_prob
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.tta_shift = tta_shift
        self.subject_cache = {}
        self.model_selection_epsilon = 1e-4

        self.criterion_train = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing).to(self.device)
        self.criterion_eval = nn.CrossEntropyLoss().to(self.device)
        self.model = None
        self._reset_model()

    @staticmethod
    def get_seq_len(n_channels: int = 30, n_times: int = 250, emb_size: int = 16) -> int:
        dummy = torch.zeros(1, 1, n_channels, n_times)
        pe = PatchEmbedding(emb_size, n_channels)
        with torch.no_grad():
            out = pe(dummy)
        return out.shape[1]

    def augment(self, x: Tensor) -> Tensor:
        x = x + torch.randn_like(x) * self.aug_noise_std
        shift = torch.randint(-self.aug_shift, self.aug_shift + 1, (1,), device=x.device).item()
        x = torch.roll(x, shift, dims=-1)
        if torch.rand(1, device=x.device).item() < self.channel_mask_prob:
            mask = (torch.rand(x.size(0), 1, x.size(2), 1, device=x.device) > self.channel_drop_prob).float()
            x = x * mask
        return x

    def eval_augment(self, x: Tensor, shift: int = 0) -> Tensor:
        if shift:
            x = torch.roll(x, shifts=shift, dims=-1)
        return x

    def get_tta_shifts(self, n_views: int):
        if n_views <= 1 or self.tta_shift <= 0:
            return [0] * max(1, n_views)
        if n_views == 2:
            return [0, self.tta_shift]

        shifts = [0]
        raw_shifts = np.linspace(-self.tta_shift, self.tta_shift, max(1, n_views - 1))
        for shift in raw_shifts:
            shift_value = int(round(float(shift)))
            if shift_value == 0:
                continue
            shifts.append(shift_value)

        while len(shifts) < n_views:
            shifts.append(0)
        return shifts[:n_views]

    @staticmethod
    def mixup(x: Tensor, y: Tensor, alpha: float = 0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam

    @staticmethod
    def _cpu_state_dict(model):
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def _build_model(self):
        model = ViT(
            emb_size=self.emb_size,
            depth=self.depth,
            n_classes=self.n_classes,
            n_channels=self.n_channels,
            seq_len=self.seq_len,
        ).to(self.device)
        if self.device.type == "cuda" and len(gpus) > 1:
            model = nn.DataParallel(model, device_ids=list(range(len(gpus)))).to(self.device)
        return model

    def _reset_model(self):
        torch.manual_seed(self.model_seed)
        np.random.seed(self.model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.model_seed)
        self.model = self._build_model()

    @torch.no_grad()
    def tta_evaluate(self, x: Tensor, n_views: int = 5) -> Tensor:
        logits_sum = None
        for shift in self.get_tta_shifts(n_views):
            x_aug = self.eval_augment(x.clone(), shift=shift)
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                _, out = self.model(x_aug)
            logits_sum = out if logits_sum is None else logits_sum + out
        return logits_sum / n_views

    @torch.no_grad()
    def batched_predict(self, x: Tensor, batch_size: int, n_tta: int = 1) -> Tensor:
        logits_list = []
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            batch = x[start:end]
            try:
                if n_tta and n_tta > 1:
                    logits = self.tta_evaluate(batch, n_views=n_tta)
                else:
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        _, logits = self.model(batch)
            except RuntimeError as exc:
                if self.device.type != "cuda" or "cuDNN algorithm" not in str(exc):
                    raise
                # Retry once with cuDNN disabled for this batch when the current
                # CUDA/cuDNN stack cannot find a valid deterministic algorithm.
                with torch.backends.cudnn.flags(enabled=False):
                    if n_tta and n_tta > 1:
                        logits = self.tta_evaluate(batch, n_views=n_tta)
                    else:
                        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                            _, logits = self.model(batch)
            logits_list.append(logits)
        return torch.cat(logits_list, dim=0)

    def _load_subject_data(self, sid: int):
        if sid in self.subject_cache:
            return self.subject_cache[sid]
        mat_file = os.path.join(self.data_dir, f"HC{sid}_1s.mat")
        mat = scipy.io.loadmat(mat_file)
        all_data = np.ascontiguousarray(mat["data"], dtype=np.float32)
        all_label = np.ascontiguousarray(mat["label"].flatten(), dtype=np.int64)
        if self.use_denoise:
            all_data = denoise_signals(all_data)
        self.subject_cache[sid] = (all_data, all_label)
        return self.subject_cache[sid]

    @staticmethod
    def _compute_normalization_stats(train_data):
        mu = train_data.mean(axis=(0, 2), keepdims=True)
        std = train_data.std(axis=(0, 2), keepdims=True) + 1e-8
        return mu, std

    @staticmethod
    def _apply_normalization(data, mu, std):
        return (data - mu) / std

    def is_better_checkpoint(
        self,
        val_loss: float,
        val_macro_acc: float,
        best_val_loss: float,
        best_val_macro_acc: float,
    ) -> bool:
        if val_macro_acc > best_val_macro_acc + self.model_selection_epsilon:
            return True
        if abs(val_macro_acc - best_val_macro_acc) <= self.model_selection_epsilon and val_loss < best_val_loss - self.model_selection_epsilon:
            return True
        if val_loss < best_val_loss - self.model_selection_epsilon:
            return abs(val_macro_acc - best_val_macro_acc) <= 5 * self.model_selection_epsilon
        if abs(val_loss - best_val_loss) <= self.model_selection_epsilon and val_macro_acc > best_val_macro_acc:
            return True
        return False

    @staticmethod
    def compute_subject_macro_accuracy(logits: Tensor, labels: Tensor, subject_ids: Tensor) -> float:
        subject_scores = []
        predictions = logits.argmax(dim=1)
        for sid in subject_ids.unique(sorted=True):
            subject_mask = subject_ids == sid
            if subject_mask.any():
                subject_acc = (predictions[subject_mask] == labels[subject_mask]).float().mean().item()
                subject_scores.append(subject_acc)
        if not subject_scores:
            return 0.0
        return float(np.mean(subject_scores))

    @staticmethod
    def _log_subject_progress(split_name: str, index: int, total: int, sid: int):
        print(f"[prepare] {split_name}: loading HC{sid} ({index}/{total})")

    def _prepare_fold_data(self, train_val_subject_ids, test_subject_ids, val_ratio=0.125, fold_seed=42):
        prepare_start = datetime.datetime.now()
        train_val_subject_ids = np.array(train_val_subject_ids)
        n_train_val = len(train_val_subject_ids)
        n_val = max(1, int(n_train_val * val_ratio))
        val_pick = np.random.RandomState(fold_seed).choice(np.arange(n_train_val), n_val, replace=False)
        val_subject_ids = train_val_subject_ids[val_pick].tolist()
        train_subject_ids = np.delete(train_val_subject_ids, val_pick).tolist()

        print(
            f"[prepare] split train/val/test subjects: "
            f"train={len(train_subject_ids)}, val={len(val_subject_ids)}, test={len(test_subject_ids)}"
        )

        train_data_list = []
        train_label_list = []
        for idx, sid in enumerate(train_subject_ids, 1):
            self._log_subject_progress("train", idx, len(train_subject_ids), int(sid))
            data, label = self._load_subject_data(int(sid))
            train_data_list.append(data)
            train_label_list.append(label)

        train_base = np.concatenate(train_data_list, axis=0)
        train_mu, train_std = self._compute_normalization_stats(train_base)
        label_concat = np.concatenate(train_label_list, axis=0)
        train_concat = self._apply_normalization(train_base, train_mu, train_std)
        train_concat = np.ascontiguousarray(train_concat[:, np.newaxis], dtype=np.float32)
        label_concat = np.ascontiguousarray(label_concat, dtype=np.int64)

        val_data_list = []
        val_label_list = []
        for idx, sid in enumerate(val_subject_ids, 1):
            self._log_subject_progress("val", idx, len(val_subject_ids), int(sid))
            data, label = self._load_subject_data(int(sid))
            data = self._apply_normalization(data, train_mu, train_std)
            val_data_list.append(data)
            val_label_list.append(label)

        test_data_list = []
        test_label_list = []
        for idx, sid in enumerate(test_subject_ids, 1):
            self._log_subject_progress("test", idx, len(test_subject_ids), int(sid))
            data, label = self._load_subject_data(int(sid))
            data = self._apply_normalization(data, train_mu, train_std)
            test_data_list.append(data)
            test_label_list.append(label)

        val_concat = np.concatenate(val_data_list, axis=0)
        val_label_concat = np.concatenate(val_label_list, axis=0)
        test_concat = np.concatenate(test_data_list, axis=0)
        test_label_concat = np.concatenate(test_label_list, axis=0)

        val_concat = np.ascontiguousarray(val_concat[:, np.newaxis], dtype=np.float32)
        test_concat = np.ascontiguousarray(test_concat[:, np.newaxis], dtype=np.float32)
        val_label_concat = np.ascontiguousarray(val_label_concat, dtype=np.int64)
        test_label_concat = np.ascontiguousarray(test_label_concat, dtype=np.int64)

        perm = np.random.permutation(len(train_concat))
        train_concat = train_concat[perm]
        label_concat = label_concat[perm]

        print(
            f"[prepare] done: train={train_concat.shape}, val={val_concat.shape}, "
            f"test={test_concat.shape}, elapsed={datetime.datetime.now() - prepare_start}"
        )

        return (
            train_concat,
            label_concat,
            val_concat,
            val_label_concat,
            test_concat,
            test_label_concat,
            train_subject_ids,
            val_subject_ids,
            np.concatenate(
                [
                    np.full(len(labels), int(sid), dtype=np.int64)
                    for sid, labels in zip(val_subject_ids, val_label_list)
                ]
            ),
        )

    def run_subject_cv(
        self,
        subject_ids,
        save_dir,
        n_epochs=200,
        batch_size=128,
        patience=60,
        min_epochs=40,
        seed=42,
        val_ratio=0.25,
        n_tta=5,
        num_workers=4,
        start_fold=1,
        end_fold=999,
        common_ckpt_name="finetuned_best.pth",
    ):
        os.makedirs(save_dir, exist_ok=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_splits = list(enumerate(kf.split(np.array(subject_ids)), 1))
        end_fold = min(end_fold, len(fold_splits))

        fold_results = []
        processed_folds = []
        best_overall_val_loss = float("inf")
        best_overall_val_acc = float("-inf")
        common_ckpt_path = os.path.join(save_dir, common_ckpt_name)

        for fold_idx, (train_val_idx, test_idx) in fold_splits:
            if fold_idx < start_fold or fold_idx > end_fold:
                continue

            self._reset_model()

            train_val_subjects = np.array(subject_ids)[train_val_idx].tolist()
            test_subjects = np.array(subject_ids)[test_idx].tolist()

            print(f"[fold {fold_idx}] preparing fold data...")

            (
                train_data,
                train_label,
                val_data,
                val_label,
                test_data,
                test_label,
                train_subjects,
                val_subjects,
                val_subject_ids_per_sample,
            ) = self._prepare_fold_data(
                train_val_subjects,
                test_subjects,
                val_ratio=val_ratio,
                fold_seed=seed + fold_idx,
            )

            print(f"\n{'=' * 60}\nFold {fold_idx}/5\n{'=' * 60}")
            print(f"Train subjects: {train_subjects}")
            print(f"Val subjects: {val_subjects}")
            print(f"Test subjects: {test_subjects}")
            print(f"Train: {len(train_label)} samples, Val: {len(val_label)}, Test: {len(test_label)}")

            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(train_data, dtype=torch.float32),
                torch.tensor(train_label, dtype=torch.long),
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(val_data, dtype=torch.float32),
                torch.tensor(val_label, dtype=torch.long),
                torch.tensor(val_subject_ids_per_sample, dtype=torch.long),
            )
            test_data_tensor = torch.tensor(test_data, dtype=torch.float32, device=self.device)
            test_label_tensor = torch.tensor(test_label, dtype=torch.long, device=self.device)

            use_pin_memory = self.device.type == "cuda"
            train_loader_kwargs = {
                "batch_size": batch_size,
                "shuffle": True,
                "num_workers": num_workers,
                "pin_memory": use_pin_memory,
            }
            val_loader_kwargs = {
                "batch_size": batch_size,
                "shuffle": False,
                "num_workers": num_workers,
                "pin_memory": use_pin_memory,
            }
            if num_workers > 0:
                train_loader_kwargs["persistent_workers"] = True
                train_loader_kwargs["prefetch_factor"] = 2
                val_loader_kwargs["persistent_workers"] = True
                val_loader_kwargs["prefetch_factor"] = 2

            train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_kwargs)
            val_loader = torch.utils.data.DataLoader(val_dataset, **val_loader_kwargs)

            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                betas=(self.b1, self.b2),
                weight_decay=self.weight_decay,
            )
            scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=n_epochs)
            ema = EMA(self.model, decay=0.999)
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

            best_val_loss = float("inf")
            best_val_macro_acc = float("-inf")
            best_val_acc = float("-inf")
            patience_counter = 0
            best_save_path = os.path.join(save_dir, f"subjectcv_fold{fold_idx}_best.pth")

            for epoch in range(n_epochs):
                self.model.train()
                train_loss = 0.0
                train_correct = 0

                for imgs, labels in train_loader:
                    imgs = imgs.to(self.device, non_blocking=use_pin_memory)
                    labels = labels.to(self.device, non_blocking=use_pin_memory)
                    imgs = self.augment(imgs)

                    use_mixup = torch.rand(1, device=self.device).item() < self.mixup_prob
                    labels_for_acc = labels

                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        if use_mixup:
                            imgs, labels_a, labels_b, lam = ExGAN.mixup(imgs, labels, alpha=self.mixup_alpha)
                            _, outputs = self.model(imgs)
                            loss = lam * self.criterion_train(outputs, labels_a) + (1 - lam) * self.criterion_train(outputs, labels_b)
                            labels_for_acc = labels_a
                        else:
                            _, outputs = self.model(imgs)
                            loss = self.criterion_train(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    ema.update()

                    train_loss += loss.item() * len(imgs)
                    if use_mixup:
                        pred = outputs.argmax(1)
                        mix_acc = lam * (pred == labels_a).float() + (1 - lam) * (pred == labels_b).float()
                        train_correct += mix_acc.sum().item()
                    else:
                        train_correct += (outputs.argmax(1) == labels_for_acc).sum().item()

                scheduler.step()
                avg_train_loss = train_loss / len(train_label)
                avg_train_acc = train_correct / len(train_label)

                self.model.eval()
                ema.apply_shadow()
                val_loss = 0.0
                val_correct = 0
                val_logits_batches = []
                val_label_batches = []
                val_subject_batches = []
                with torch.no_grad():
                    for v_imgs, v_labels, v_subjects in val_loader:
                        v_imgs = v_imgs.to(self.device, non_blocking=use_pin_memory)
                        v_labels = v_labels.to(self.device, non_blocking=use_pin_memory)
                        v_subjects = v_subjects.to(self.device, non_blocking=use_pin_memory)
                        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                            _, v_outputs = self.model(v_imgs)
                            v_loss = self.criterion_eval(v_outputs, v_labels)
                        val_loss += v_loss.item() * len(v_imgs)
                        val_correct += (v_outputs.argmax(1) == v_labels).sum().item()
                        val_logits_batches.append(v_outputs.detach())
                        val_label_batches.append(v_labels.detach())
                        val_subject_batches.append(v_subjects.detach())
                ema.restore()

                avg_val_loss = val_loss / len(val_label)
                avg_val_acc = val_correct / len(val_label)
                val_logits_all = torch.cat(val_logits_batches, dim=0)
                val_labels_all = torch.cat(val_label_batches, dim=0)
                val_subjects_all = torch.cat(val_subject_batches, dim=0)
                avg_val_macro_acc = self.compute_subject_macro_accuracy(val_logits_all, val_labels_all, val_subjects_all)

                print(
                    f"Epoch {epoch + 1:3d}/{n_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f} Macro: {avg_val_macro_acc:.4f}"
                )

                if self.is_better_checkpoint(avg_val_loss, avg_val_macro_acc, best_val_loss, best_val_macro_acc):
                    best_val_loss = avg_val_loss
                    best_val_macro_acc = avg_val_macro_acc
                    best_val_acc = avg_val_acc
                    ema.apply_shadow()
                    torch.save(self._cpu_state_dict(self.model), best_save_path)
                    ema.restore()
                    patience_counter = 0
                    print(
                        f"new best: loss={best_val_loss:.4f}, "
                        f"macro={best_val_macro_acc:.4f}, acc={best_val_acc:.4f}, save to: {best_save_path}"
                    )
                else:
                    patience_counter += 1

                if epoch + 1 >= min_epochs and patience_counter >= patience:
                    print(f"Early stopping after {patience} epochs without val loss improvement.")
                    break

            best_state = torch.load(best_save_path, map_location="cpu")
            self.model.load_state_dict(best_state, strict=True)

            self.model.eval()
            with torch.no_grad():
                cls_out = self.batched_predict(test_data_tensor, batch_size=batch_size, n_tta=n_tta)

            y_pred = cls_out.argmax(dim=1)
            test_acc = (y_pred == test_label_tensor).float().mean().item()
            fold_results.append(test_acc)
            processed_folds.append(fold_idx)

            if self.is_better_checkpoint(best_val_loss, best_val_macro_acc, best_overall_val_loss, best_overall_val_acc):
                best_overall_val_loss = best_val_loss
                best_overall_val_acc = best_val_macro_acc
                torch.save(best_state, common_ckpt_path)

            print(
                f"Fold {fold_idx} Test Acc: {test_acc * 100:.2f}% "
                f"(best val loss: {best_val_loss:.4f}, best val macro: {best_val_macro_acc * 100:.2f}%, "
                f"best val acc: {best_val_acc * 100:.2f}%)"
            )

        if not fold_results:
            print("No folds were run.")
            return []

        mean_acc = float(np.mean(fold_results))
        std_acc = float(np.std(fold_results))

        print(f"\n{'=' * 60}")
        print(f"CV Results: {[f'{acc * 100:.2f}' for acc in fold_results]}")
        print(f"Mean: {mean_acc * 100:.2f}% +- {std_acc * 100:.2f}%")
        print(f"Best shared checkpoint: {common_ckpt_path}")
        print(f"{'=' * 60}")

        with open(os.path.join(save_dir, "CV_RESULTS.txt"), "w", encoding="utf-8") as f:
            f.write("EEG-Conformer 5-Fold CV Results\n")
            f.write(f"Folds run: {processed_folds}\n")
            f.write(f"Fold accuracies: {fold_results}\n")
            f.write(f"Mean: {mean_acc * 100:.2f}% +- {std_acc * 100:.2f}%\n")
            f.write(f"Best shared checkpoint: {common_ckpt_path}\n")

        return fold_results


def parse_args():
    parser = argparse.ArgumentParser(description="EEG-Conformer subject-level 5-fold CV training.")
    parser.add_argument("--data_dir", type=str, default="./EEG-Conformer/data/processed_normal/")
    parser.add_argument("--save_dir", type=str, default="./EEG-Conformer/last_params/")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--emb_size", type=int, default=40)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_seed", type=int, default=-1)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--n_tta", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_denoise", dest="use_denoise", action="store_true")
    parser.add_argument("--no_denoise", dest="use_denoise", action="store_false")
    parser.set_defaults(use_denoise=True)
    parser.add_argument("--disable_amp", action="store_true", default=False)
    parser.add_argument("--aug_noise_std", type=float, default=0.02)
    parser.add_argument("--aug_shift", type=int, default=12)
    parser.add_argument("--channel_mask_prob", type=float, default=0.15)
    parser.add_argument("--channel_drop_prob", type=float, default=0.08)
    parser.add_argument("--mixup_prob", type=float, default=0.2)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--tta_shift", type=int, default=4)
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--end_fold", type=int, default=1)
    parser.add_argument("--common_ckpt_name", type=str, default="finetuned_best.pth")
    return parser.parse_args()


def configure_gpus(gpu_arg: str):
    global gpus
    gpu_values = [item.strip() for item in str(gpu_arg).split(",") if item.strip()]
    gpus = [int(item) for item in gpu_values] if gpu_values else [0]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in gpus)


def read_data(data_dir: str):
    subject_ids = sorted(
        [
            int(os.path.basename(f).replace("HC", "").replace("_1s.mat", ""))
            for f in glob.glob(os.path.join(data_dir, "HC*_1s.mat"))
        ]
    )
    return subject_ids


def main():
    args = parse_args()
    configure_gpus(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    subject_ids = read_data(args.data_dir)
    n_subjects = len(subject_ids)
    if n_subjects == 0:
        print(f"No data found under {args.data_dir}")
        return

    print(f"Using device: {device}")
    print(f"Found {n_subjects} subjects: {subject_ids}")

    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=250, emb_size=args.emb_size)
    print(f"[Info] seq_len = {seq_len}")

    start_time = datetime.datetime.now()
    model_seed = args.model_seed if args.model_seed >= 0 else args.seed

    trainer = ExGAN(
        data_dir=args.data_dir,
        seq_len=seq_len,
        depth=args.depth,
        emb_size=args.emb_size,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        model_seed=model_seed,
        use_denoise=args.use_denoise,
        amp_enabled=not args.disable_amp,
        aug_noise_std=args.aug_noise_std,
        aug_shift=args.aug_shift,
        channel_mask_prob=args.channel_mask_prob,
        channel_drop_prob=args.channel_drop_prob,
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing,
        tta_shift=args.tta_shift,
    )
    trainer.run_subject_cv(
        subject_ids=subject_ids,
        save_dir=args.save_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        min_epochs=args.min_epochs,
        seed=args.seed,
        val_ratio=args.val_ratio,
        n_tta=args.n_tta,
        num_workers=args.num_workers,
        start_fold=args.start_fold,
        end_fold=args.end_fold,
        common_ckpt_name=args.common_ckpt_name,
    )
    print(f"Total elapsed: {datetime.datetime.now() - start_time}")


if __name__ == "__main__":
    main()
