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


class BandTokenAggregator(nn.Module):
    def __init__(self, in_channels: int, emb_size: int):
        super().__init__()
        hidden_channels = max(16, in_channels // 2)
        self.attn_logits = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, (1, 1), bias=False),
            nn.GroupNorm(pick_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, (1, 1), bias=True),
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
        tokens = (attn * values).sum(dim=2).transpose(1, 2)
        return self.token_norm(self.token_dropout(tokens))


class BandPowerEncoder(nn.Module):
    def __init__(self, sampling_rate: int = 250):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.band_limits = (
            (1.0, 4.0),
            (4.0, 8.0),
            (8.0, 13.0),
            (13.0, 30.0),
            (30.0, 45.0),
        )
        self.num_bands = len(self.band_limits)

    def forward(self, x: Tensor) -> Tensor:
        power = torch.abs(torch.fft.rfft(x.float(), dim=-1)).pow(2)[..., 1:]
        total_power = power.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        freqs = torch.linspace(
            0.0,
            self.sampling_rate / 2,
            power.shape[-1] + 1,
            device=power.device,
            dtype=power.dtype,
        )[1:]
        absolute_maps = []
        relative_maps = []
        for low_hz, high_hz in self.band_limits:
            mask = (freqs >= low_hz) & (freqs < high_hz)
            if mask.any():
                band_energy = power[..., mask].sum(dim=-1)
            else:
                center_idx = int(torch.argmin((freqs - (low_hz + high_hz) * 0.5).abs()).item())
                band_energy = power[..., center_idx : center_idx + 1].sum(dim=-1)
            absolute_maps.append(torch.log1p(band_energy).unsqueeze(-1))
            relative_maps.append((band_energy / total_power.squeeze(-1)).unsqueeze(-1))

        absolute_map = torch.cat(absolute_maps, dim=-1)
        relative_map = torch.cat(relative_maps, dim=-1)
        absolute_map = (absolute_map - absolute_map.mean(dim=-1, keepdim=True)) / (
            absolute_map.std(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        relative_map = (relative_map - relative_map.mean(dim=-1, keepdim=True)) / (
            relative_map.std(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        return torch.cat([absolute_map, relative_map], dim=1)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40, n_channels: int = 30, sampling_rate: int = 250):
        super().__init__()
        branch_channels = 20
        temporal_channels = branch_channels * 3
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

        self.band_encoder = BandPowerEncoder(sampling_rate=sampling_rate)
        self.band_stem = nn.Sequential(
            nn.Conv2d(2, feat_channels, (1, 1), stride=(1, 1), bias=False),
            nn.GroupNorm(feat_groups, feat_channels),
            nn.SiLU(),
        )
        self.band_context = nn.Sequential(
            nn.Conv2d(
                feat_channels,
                feat_channels,
                (1, 3),
                stride=(1, 1),
                padding=(0, 1),
                groups=feat_channels,
                bias=False,
            ),
            nn.GroupNorm(feat_groups, feat_channels),
            nn.SiLU(),
        )
        self.band_reweight = SqueezeExcite2d(feat_channels)
        self.band_refine = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, (1, 1), stride=(1, 1), bias=False),
            nn.GroupNorm(feat_groups, feat_channels),
            nn.SiLU(),
        )
        self.band_tokens = BandTokenAggregator(feat_channels, emb_size)
        self.time_token_type = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)
        self.band_token_type = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)
        self.band_embedding = nn.Parameter(torch.randn(1, self.band_encoder.num_bands, emb_size) * 0.02)

    def forward(self, x: Tensor):
        x = self.input_norm(x)
        time_feat = torch.cat([branch(x) for branch in self.temporal_branches], dim=1)
        time_feat = self.temporal_fuse(time_feat)
        time_feat = self.temporal_context(time_feat)
        time_feat = self.channel_reweight(time_feat)
        time_feat = self.temporal_pool(time_feat)
        time_tokens = self.spatial_aggregator(time_feat)
        time_tokens = time_tokens + self.time_token_type

        band_feat = self.band_encoder(x)
        band_feat = self.band_stem(band_feat)
        band_feat = self.band_context(band_feat)
        band_feat = self.band_reweight(band_feat)
        band_feat = self.band_refine(band_feat)
        band_tokens = self.band_tokens(band_feat).to(time_tokens.dtype)
        band_tokens = band_tokens + self.band_token_type + self.band_embedding

        return time_tokens, band_tokens


class CrossAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.scale = (emb_size // num_heads) ** -0.5
        self.query_proj = nn.Linear(emb_size, emb_size)
        self.key_proj = nn.Linear(emb_size, emb_size)
        self.value_proj = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(emb_size, emb_size)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        queries = rearrange(self.query_proj(query), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.key_proj(context), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.value_proj(context), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum("bhqd,bhkd->bhqk", queries, keys) * self.scale
        att = self.att_drop(torch.softmax(energy, dim=-1))
        out = torch.einsum("bhqk,bhkd->bhqd", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.out_proj(out)


class CrossAttentionBlock(nn.Module):
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
        self.query_norm = nn.LayerNorm(emb_size)
        self.context_norm = nn.LayerNorm(emb_size)
        self.cross_attn = CrossAttention(emb_size, num_heads, drop_p)
        self.cross_drop = nn.Dropout(drop_p)
        self.cross_scale = LayerScale(emb_size)
        self.cross_path = DropPath(drop_path)
        self.ffn_norm = nn.LayerNorm(emb_size)
        self.ffn = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self.ffn_drop = nn.Dropout(drop_p)
        self.ffn_scale = LayerScale(emb_size)
        self.ffn_path = DropPath(drop_path)
        self.final_norm = nn.LayerNorm(emb_size)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        x = x + self.cross_path(self.cross_scale(self.cross_drop(self.cross_attn(self.query_norm(x), self.context_norm(context)))))
        x = x + self.ffn_path(self.ffn_scale(self.ffn_drop(self.ffn(self.ffn_norm(x)))))
        return self.final_norm(x)


class TimeFrequencyFusion(nn.Module):
    def __init__(self, emb_size: int, num_heads: int = 4, drop_p: float = 0.1):
        super().__init__()
        self.time_to_band = CrossAttentionBlock(emb_size, num_heads=num_heads, drop_p=drop_p)
        self.band_to_time = CrossAttentionBlock(emb_size, num_heads=num_heads, drop_p=drop_p)
        self.time_gate = nn.Parameter(torch.full((1, 1, emb_size), 0.5))
        self.band_gate = nn.Parameter(torch.full((1, 1, emb_size), 0.5))
        self.time_norm = nn.LayerNorm(emb_size)
        self.band_norm = nn.LayerNorm(emb_size)

    def forward(self, time_tokens: Tensor, band_tokens: Tensor):
        time_update = self.time_to_band(time_tokens, band_tokens)
        band_update = self.band_to_time(band_tokens, time_tokens)
        time_tokens = self.time_norm(time_tokens + torch.tanh(self.time_gate) * time_update)
        band_tokens = self.band_norm(band_tokens + torch.tanh(self.band_gate) * band_update)
        return time_tokens, band_tokens


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambda_value: float):
        ctx.lambda_value = lambda_value
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.lambda_value * grad_output, None


class GradientReversal(nn.Module):
    def forward(self, x: Tensor, lambda_value: float) -> Tensor:
        return GradientReversalFunction.apply(x, lambda_value)


class SubjectDiscriminator(nn.Module):
    def __init__(self, in_features: int, n_subjects: int):
        super().__init__()
        hidden_size = max(64, in_features // 2)
        self.grl = GradientReversal()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, n_subjects),
        )

    def forward(self, x: Tensor, lambda_value: float) -> Tensor:
        x = self.grl(x, lambda_value)
        return self.net(x)


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
    def __init__(
        self,
        emb_size: int,
        depth: int,
        n_classes: int = 2,
        n_channels: int = 30,
        seq_len: int = 11,
        sampling_rate: int = 250,
        n_subjects: int = 0,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels, sampling_rate=sampling_rate)
        self.fusion = TimeFrequencyFusion(emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)
        self.pos_encoder = ConvPositionalEncoding(emb_size)
        self.pos_drop = nn.Dropout(0.1)
        self.transformer = TransformerEncoder(depth, emb_size)
        self.cls_head = ClassificationHead(emb_size, n_classes)
        self.subject_head = SubjectDiscriminator(emb_size * 3, n_subjects) if n_subjects > 1 else None

    def forward(self, x: Tensor, subject_lambda: float = 0.0):
        time_tokens, band_tokens = self.patch_embedding(x)
        time_tokens, band_tokens = self.fusion(time_tokens, band_tokens)
        x = torch.cat([time_tokens, band_tokens], dim=1)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(self.pos_encoder(x))
        x = self.transformer(x)
        feat, logits = self.cls_head(x)
        subject_logits = None
        if self.subject_head is not None and subject_lambda > 0:
            subject_logits = self.subject_head(feat, subject_lambda)
        return feat, logits, subject_logits


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
        all_subject_ids,
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
        sampling_rate: int = 250,
        subject_adv_weight: float = 0.05,
        subject_adv_warmup: int = 8,
    ):
        self.n_channels = 30
        self.n_times = 250
        self.n_classes = 2
        self.sampling_rate = sampling_rate
        self.lr = lr
        self.b1, self.b2 = 0.9, 0.999
        self.weight_decay = weight_decay
        self.data_dir = data_dir
        self.all_subject_ids = [int(sid) for sid in all_subject_ids]
        self.subject_id_to_index = {sid: idx for idx, sid in enumerate(self.all_subject_ids)}
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
        self.subject_adv_weight = subject_adv_weight
        self.subject_adv_warmup = subject_adv_warmup
        self.subject_cache = {}
        self.model_selection_epsilon = 1e-4

        self.criterion_train = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing).to(self.device)
        self.criterion_eval = nn.CrossEntropyLoss().to(self.device)
        self.criterion_subject = nn.CrossEntropyLoss().to(self.device)
        self.model = None
        self._reset_model()

    @staticmethod
    def get_seq_len(
        n_channels: int = 30,
        n_times: int = 250,
        emb_size: int = 16,
        sampling_rate: int = 250,
    ) -> int:
        dummy = torch.zeros(1, 1, n_channels, n_times)
        pe = PatchEmbedding(emb_size, n_channels, sampling_rate=sampling_rate)
        with torch.no_grad():
            time_tokens, band_tokens = pe(dummy)
        return time_tokens.shape[1] + band_tokens.shape[1]

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
            sampling_rate=self.sampling_rate,
            n_subjects=len(self.subject_id_to_index),
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
                _, out, _ = self.model(x_aug)
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
                        _, logits, _ = self.model(batch)
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
                            _, logits, _ = self.model(batch)
            logits_list.append(logits)
        return torch.cat(logits_list, dim=0)

    def get_subject_adv_lambda(self, epoch: int, total_epochs: int) -> float:
        if self.subject_adv_weight <= 0 or len(self.subject_id_to_index) <= 1:
            return 0.0
        if epoch + 1 <= self.subject_adv_warmup:
            return 0.0
        progress = (epoch + 1 - self.subject_adv_warmup) / max(1, total_epochs - self.subject_adv_warmup)
        progress = float(np.clip(progress, 0.0, 1.0))
        return 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0

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
        train_subject_index_concat = np.concatenate(
            [
                np.full(len(labels), self.subject_id_to_index[int(sid)], dtype=np.int64)
                for sid, labels in zip(train_subject_ids, train_label_list)
            ]
        )
        train_concat = self._apply_normalization(train_base, train_mu, train_std)
        train_concat = np.ascontiguousarray(train_concat[:, np.newaxis], dtype=np.float32)
        label_concat = np.ascontiguousarray(label_concat, dtype=np.int64)
        train_subject_index_concat = np.ascontiguousarray(train_subject_index_concat, dtype=np.int64)

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
        train_subject_index_concat = train_subject_index_concat[perm]

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
            train_subject_index_concat,
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
                train_subject_index,
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
                torch.tensor(train_subject_index, dtype=torch.long),
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
                train_adv_loss = 0.0
                train_correct = 0
                subject_lambda = self.get_subject_adv_lambda(epoch, n_epochs)
                subject_loss_weight = self.subject_adv_weight * subject_lambda

                for imgs, labels, subject_labels in train_loader:
                    imgs = imgs.to(self.device, non_blocking=use_pin_memory)
                    labels = labels.to(self.device, non_blocking=use_pin_memory)
                    subject_labels = subject_labels.to(self.device, non_blocking=use_pin_memory)
                    imgs = self.augment(imgs)

                    use_mixup = torch.rand(1, device=self.device).item() < self.mixup_prob
                    labels_for_acc = labels

                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        if use_mixup:
                            imgs, labels_a, labels_b, lam = ExGAN.mixup(imgs, labels, alpha=self.mixup_alpha)
                            _, outputs, _ = self.model(imgs)
                            cls_loss = lam * self.criterion_train(outputs, labels_a) + (1 - lam) * self.criterion_train(outputs, labels_b)
                            adv_loss = outputs.new_zeros(())
                            labels_for_acc = labels_a
                        else:
                            _, outputs, subject_logits = self.model(imgs, subject_lambda=subject_lambda)
                            cls_loss = self.criterion_train(outputs, labels)
                            if subject_logits is not None and subject_loss_weight > 0:
                                adv_loss = self.criterion_subject(subject_logits, subject_labels)
                            else:
                                adv_loss = outputs.new_zeros(())
                        loss = cls_loss + subject_loss_weight * adv_loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    ema.update()

                    train_loss += loss.item() * len(imgs)
                    train_adv_loss += adv_loss.item() * len(imgs)
                    if use_mixup:
                        pred = outputs.argmax(1)
                        mix_acc = lam * (pred == labels_a).float() + (1 - lam) * (pred == labels_b).float()
                        train_correct += mix_acc.sum().item()
                    else:
                        train_correct += (outputs.argmax(1) == labels_for_acc).sum().item()

                scheduler.step()
                avg_train_loss = train_loss / len(train_label)
                avg_train_adv_loss = train_adv_loss / len(train_label)
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
                            _, v_outputs, _ = self.model(v_imgs)
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
                    f"Train Loss: {avg_train_loss:.4f} Adv: {avg_train_adv_loss:.4f} Acc: {avg_train_acc:.4f} | "
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
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--training_mode", choices=["baseline", "full"], default="baseline")
    pre_args, _ = pre_parser.parse_known_args()

    mode_defaults = {
        "baseline": {
            "batch_size": 64,
            "epochs": 120,
            "patience": 30,
            "min_epochs": 50,
            "val_ratio": 0.125,
            "n_tta": 1,
            "use_denoise": False,
            "aug_noise_std": 0.01,
            "aug_shift": 8,
            "channel_mask_prob": 0.05,
            "channel_drop_prob": 0.0,
            "mixup_prob": 0.0,
            "mixup_alpha": 0.2,
            "label_smoothing": 0.02,
            "subject_adv_weight": 0.0,
            "subject_adv_warmup": 0,
        },
        "full": {
            "batch_size": 64,
            "epochs": 120,
            "patience": 20,
            "min_epochs": 40,
            "val_ratio": 0.25,
            "n_tta": 1,
            "use_denoise": True,
            "aug_noise_std": 0.02,
            "aug_shift": 12,
            "channel_mask_prob": 0.15,
            "channel_drop_prob": 0.08,
            "mixup_prob": 0.2,
            "mixup_alpha": 0.2,
            "label_smoothing": 0.05,
            "subject_adv_weight": 0.05,
            "subject_adv_warmup": 8,
        },
    }[pre_args.training_mode]

    parser = argparse.ArgumentParser(description="EEG-Conformer subject-level 5-fold CV training.")
    parser.add_argument("--training_mode", choices=["baseline", "full"], default="full")
    parser.add_argument("--data_dir", type=str, default="./EEG-Conformer/data/processed_normal/")
    parser.add_argument("--save_dir", type=str, default="./EEG-Conformer/last_params/")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--emb_size", type=int, default=40)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=mode_defaults["batch_size"])
    parser.add_argument("--epochs", type=int, default=mode_defaults["epochs"])
    parser.add_argument("--patience", type=int, default=mode_defaults["patience"])
    parser.add_argument("--min_epochs", type=int, default=mode_defaults["min_epochs"])
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_seed", type=int, default=-1)
    parser.add_argument("--val_ratio", type=float, default=mode_defaults["val_ratio"])
    parser.add_argument("--n_tta", type=int, default=mode_defaults["n_tta"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_denoise", dest="use_denoise", action="store_true")
    parser.add_argument("--no_denoise", dest="use_denoise", action="store_false")
    parser.set_defaults(use_denoise=mode_defaults["use_denoise"])
    parser.add_argument("--disable_amp", action="store_true", default=False)
    parser.add_argument("--aug_noise_std", type=float, default=mode_defaults["aug_noise_std"])
    parser.add_argument("--aug_shift", type=int, default=mode_defaults["aug_shift"])
    parser.add_argument("--channel_mask_prob", type=float, default=mode_defaults["channel_mask_prob"])
    parser.add_argument("--channel_drop_prob", type=float, default=mode_defaults["channel_drop_prob"])
    parser.add_argument("--mixup_prob", type=float, default=mode_defaults["mixup_prob"])
    parser.add_argument("--mixup_alpha", type=float, default=mode_defaults["mixup_alpha"])
    parser.add_argument("--label_smoothing", type=float, default=mode_defaults["label_smoothing"])
    parser.add_argument("--tta_shift", type=int, default=4)
    parser.add_argument("--sampling_rate", type=int, default=250)
    parser.add_argument("--subject_adv_weight", type=float, default=mode_defaults["subject_adv_weight"])
    parser.add_argument("--subject_adv_warmup", type=int, default=mode_defaults["subject_adv_warmup"])
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

    seq_len = ExGAN.get_seq_len(
        n_channels=30,
        n_times=250,
        emb_size=args.emb_size,
        sampling_rate=args.sampling_rate,
    )
    print(f"[Info] seq_len = {seq_len}")
    print(
        f"[Mode] {args.training_mode} | "
        f"val_ratio={args.val_ratio} | denoise={args.use_denoise} | "
        f"mixup={args.mixup_prob} | mask={args.channel_mask_prob} | drop={args.channel_drop_prob} | "
        f"label_smoothing={args.label_smoothing} | subject_adv={args.subject_adv_weight} | "
        f"n_tta={args.n_tta} | patience={args.patience} | min_epochs={args.min_epochs}"
    )

    start_time = datetime.datetime.now()
    model_seed = args.model_seed if args.model_seed >= 0 else args.seed

    trainer = ExGAN(
        data_dir=args.data_dir,
        all_subject_ids=subject_ids,
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
        sampling_rate=args.sampling_rate,
        subject_adv_weight=args.subject_adv_weight,
        subject_adv_warmup=args.subject_adv_warmup,
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
