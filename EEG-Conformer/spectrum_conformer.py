import os, random, datetime, time, glob
import scipy.io
import numpy as np
from scipy.signal import welch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.backends import cudnn
from einops import rearrange
from einops.layers.torch import Rearrange

cudnn.benchmark     = False
cudnn.deterministic = True

gpus = [0]
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

def denoise_signals(all_data):
    """物理去噪: 幅值截断 + CAR 空间滤波"""
    std_val = np.std(all_data)
    all_data = np.clip(all_data, -3 * std_val, 3 * std_val)
    common_mode_noise = np.mean(all_data, axis=1, keepdims=True)
    return all_data - common_mode_noise


class LayerScale(nn.Module):
    """LayerScale (CaiT) — 稳定深层 Transformer 训练"""
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 24, n_channels: int = 30,
                 in_channels: int = 1):
        super().__init__()
        # 第1步：逐频率点独立线性映射 — 每个 Hz 只和自己做映射，不混合频率
        self.pointwise = nn.Conv2d(in_channels, 20, (1, 1), stride=(1, 1))

        # 第2步：多尺度频率卷积 — 在已映射的嵌入空间中捕获局部频谱模式
        # narrow (3 Hz): 捕获窄带模式如 α 峰 (~10Hz)、δ 峰 (~3Hz)
        self.freq_narrow = nn.Sequential(
            nn.Conv2d(20, 8, (1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(8),
            nn.ELU(),
        )
        # medium (5 Hz): 捕获中带模式如 β 节律 (~15-30Hz)
        self.freq_medium = nn.Sequential(
            nn.Conv2d(20, 8, (1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
        )
        # wide (11 Hz): 捕获宽带跨节律耦合
        self.freq_wide = nn.Sequential(
            nn.Conv2d(20, 8, (1, 11), stride=(1, 1), padding=(0, 5)),
            nn.BatchNorm2d(8),
            nn.ELU(),
        )

        # 合并通道: pointwise(20) + narrow(8) + medium(8) + wide(8) = 44
        in_channels = 44

        # 第3步：跨电极空间卷积 — 混合 30 个电极的信息
        self.spatial = nn.Conv2d(in_channels, 40, (n_channels, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(40)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.3)

        # 第4步：投影到 Transformer 嵌入维度
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e h w -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # (B, 1, 30, 50) → (B, 20, 30, 50)
        pw = self.pointwise(x)

        # 多尺度分支，各自保持 (30, 50) 空间维度
        narrow = self.freq_narrow(pw)  # (B, 8, 30, 50)
        medium = self.freq_medium(pw)  # (B, 8, 30, 50)
        wide   = self.freq_wide(pw)    # (B, 8, 30, 50)

        # 拼接逐点特征与多尺度特征 → (B, 44, 30, 50)
        x = torch.cat([pw, narrow, medium, wide], dim=1)

        # 空间卷积压缩电极维度 → (B, 40, 1, 50)
        x = self.spatial(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.dropout(x)

        # 投影 → (B, 50, emb_size)
        return self.projection(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.emb_size   = emb_size
        self.num_heads  = num_heads
        self.keys       = nn.Linear(emb_size, emb_size)
        self.queries    = nn.Linear(emb_size, emb_size)
        self.values     = nn.Linear(emb_size, emb_size)
        self.att_drop   = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        if mask is not None:
            energy = energy.masked_fill_(~mask, torch.finfo(torch.float32).min)
        att = self.att_drop(F.softmax(energy / (self.emb_size / self.num_heads) ** 0.5, dim=-1))
        out = torch.einsum('bhal, bhlv -> bhav', att, v)
        return self.projection(rearrange(out, "b h n d -> b n (h d)"))

class DropPath(nn.Module):
    """Stochastic Depth — 训练时随机丢弃整条残差分支"""
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

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int, num_heads: int = 4,
                 drop_p: float = 0.3, forward_expansion: int = 4,
                 forward_drop_p: float = 0.3, drop_path: float = 0.0):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
                LayerScale(emb_size),
                DropPath(drop_path),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion,
                                 drop_p=forward_drop_p),
                nn.Dropout(drop_p),
                LayerScale(emb_size),
                DropPath(drop_path),
            )),
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int, drop_path_max: float = 0.2):
        drop_path_rates = [drop_path_max * i / (depth - 1) for i in range(depth)] if depth > 1 else [0.0]
        super().__init__(*[
            TransformerEncoderBlock(emb_size, drop_path=drop_path_rates[i])
            for i in range(depth)
        ])

class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(emb_size * 2, n_classes),
        )

    def forward(self, x: Tensor):
        feat = x[:, 0]  # CLS token: (B, emb_size)
        return feat, self.fc(feat)


class ViT(nn.Module):
    def __init__(self, emb_size: int = 24, depth: int = 2,
                 n_classes: int = 2, n_channels: int = 30, seq_len: int = 50,
                 in_channels: int = 1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels, in_channels)
        self.cls_token       = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)
        self.pos_embedding   = nn.Parameter(torch.randn(1, seq_len + 1, emb_size) * 0.02)
        self.transformer     = TransformerEncoder(depth, emb_size)
        self.cls_head        = ClassificationHead(emb_size, n_classes)

    def forward(self, x: Tensor):
        x = self.patch_embedding(x)                      # (B, seq_len, emb_size)
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)    # (B, 1, emb_size)
        x = torch.cat([cls_tokens, x], dim=1)            # (B, 1+seq_len, emb_size)
        x = x + self.pos_embedding
        x = self.transformer(x)
        return self.cls_head(x)


class EMA:
    """指数移动平均 — 验证时使用参数滑动平均，通常带来 0.5-1% 提升"""
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
                self.shadow[name].copy_(
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


class ExGAN:
    def __init__(self, data_dir: str, seq_len: int, depth: int, emb_size: int,
                 psd_method: str = 'welch', detrend_spectrum: bool = False,
                 include_phase: bool = False,
                 freq_mask_width: int = 8, freq_mask_n: int = 2):
        self.n_channels = 30
        self.n_times    = 250
        self.n_classes  = 2
        self.lr         = 0.0002
        self.b1, self.b2 = 0.5, 0.999
        self.data_dir   = data_dir
        self.seq_len    = seq_len
        self.depth      = depth
        self.emb_size   = emb_size
        self.psd_method      = psd_method
        self.detrend_spectrum = detrend_spectrum
        self.include_phase   = include_phase
        self.freq_mask_width = freq_mask_width
        self.freq_mask_n     = freq_mask_n

        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()

        self.model = ViT(
            emb_size=self.emb_size, depth=self.depth, n_classes=self.n_classes,
            n_channels=self.n_channels, seq_len=seq_len,
            in_channels=2 if include_phase else 1
        ).cuda()
        self.model = nn.DataParallel(
            self.model, device_ids=list(range(len(gpus)))
        ).cuda()

    @staticmethod
    def extract_log_psd(data, fs=250, method='welch', nperseg=125, noverlap=62,
                        detrend_spectrum=False, include_phase=False):
        trials, channels, timepoints = data.shape

        if method == 'welch':
            freqs, psd = welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap,
                               window='hamming', axis=2)
        else:
            fft_data = np.fft.fft(data, axis=2)
            psd = (np.abs(fft_data) ** 2) / timepoints
            freqs = np.fft.fftfreq(timepoints, 1 / fs)

        idx = np.where((freqs >= 1) & (freqs <= 50))[0]
        psd_features = psd[:, :, idx]
        freqs_sel = freqs[idx]

        log_psd = np.log10(psd_features + 1e-8)

        if detrend_spectrum:
            log_f = np.log10(freqs_sel)
            for t in range(trials):
                for c in range(channels):
                    slope, intercept = np.polyfit(log_f, log_psd[t, c, :], 1)
                    log_psd[t, c, :] -= (slope * log_f + intercept)

        if include_phase:
            fft_data = np.fft.fft(data, axis=2)
            phase = np.angle(fft_data[:, :, idx])
            log_psd = np.stack([log_psd, phase], axis=1)
            return log_psd  # (trials, 2, channels, freq)

        return log_psd  # (trials, channels, freq)

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
    def freq_mask(data: np.ndarray, max_width: int = 8,
                  n_masks: int = 2) -> np.ndarray:
        n_freq = data.shape[-1]
        out = data.copy()
        for _ in range(n_masks):
            w = np.random.randint(1, max_width + 1)
            s = np.random.randint(0, n_freq - w + 1)
            out[..., s:s + w] = 0.0
        return out

    @staticmethod
    def augment(x: Tensor, freq_mask_width: int = 8, freq_mask_n: int = 2,
                noise_std: float = 0.01, channel_drop_p: float = 0.0) -> Tensor:
        """频域在线增强: 频率掩码 + 高斯噪声 (仅训练时调用)"""
        # 1. 频率掩码 (SpecAugment 风格)
        if freq_mask_width > 0 and freq_mask_n > 0:
            x = x.clone()
            n_freq = x.shape[-1]
            for _ in range(freq_mask_n):
                w = torch.randint(1, freq_mask_width + 1, (1,)).item()
                s = torch.randint(0, n_freq - w + 1, (1,)).item()
                x[..., s:s + w] = 0.0
        # 2. 高斯噪声
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        return x

    @staticmethod
    def get_n_freqs(fs: int = 250, method: str = 'welch', nperseg: int = 125,
                    n_times: int = 250, freq_min: int = 1, freq_max: int = 50) -> int:
        if method == 'welch':
            resolution = fs / nperseg
            n_total = nperseg // 2 + 1
        else:
            resolution = fs / n_times
            n_total = n_times // 2 + 1
        freqs = np.arange(n_total) * resolution
        return int(np.sum((freqs >= freq_min) & (freqs <= freq_max)))

    @staticmethod
    def get_seq_len(n_channels: int = 30, n_times: int = 250,
                    emb_size: int = 16, in_channels: int = 1) -> int:
        dummy = torch.zeros(1, in_channels, n_channels, n_times)
        pe    = PatchEmbedding(emb_size, n_channels, in_channels)
        with torch.no_grad():
            out = pe(dummy)
        return out.shape[1]

    def _get_train_test_data(self, sid: int):
        mat_file = os.path.join(self.data_dir, f'HC{sid}_1s.mat')
        mat      = scipy.io.loadmat(mat_file)

        all_data  = np.ascontiguousarray(mat['data'],            dtype=np.float32)
        all_label = np.ascontiguousarray(mat['label'].flatten(), dtype=np.int64)

        all_data = denoise_signals(all_data)

        all_data = self.extract_log_psd(
            all_data, fs=250, method=self.psd_method,
            detrend_spectrum=self.detrend_spectrum,
            include_phase=self.include_phase)

        train_idx_list, test_idx_list = [], []
        for cls in [0, 1]:
            cls_idx   = np.where(all_label == cls)[0]
            rng       = np.random.RandomState(sid)
            cls_idx   = cls_idx[rng.permutation(len(cls_idx))]
            
            split_point = int(len(cls_idx) * 0.8)
            train_idx_list.append(cls_idx[:split_point])
            test_idx_list.append(cls_idx[split_point:])

        train_idx = np.concatenate(train_idx_list)
        test_idx  = np.concatenate(test_idx_list)

        train_data,  train_label = all_data[train_idx], all_label[train_idx]
        test_data,   test_label  = all_data[test_idx],  all_label[test_idx]

        if self.include_phase:
            # magnitude (通道0) 和 phase (通道1) 范围不同，分别归一化
            mu_m  = train_data[:, 0].mean(axis=(0, 1), keepdims=True)
            std_m = train_data[:, 0].std(axis=(0, 1), keepdims=True) + 1e-8
            mu_p  = train_data[:, 1].mean(axis=(0, 1), keepdims=True)
            std_p = train_data[:, 1].std(axis=(0, 1), keepdims=True) + 1e-8
            train_data[:, 0] = (train_data[:, 0] - mu_m) / std_m
            test_data[:, 0]  = (test_data[:, 0]  - mu_m) / std_m
            train_data[:, 1] = (train_data[:, 1] - mu_p) / std_p
            test_data[:, 1]  = (test_data[:, 1]  - mu_p) / std_p
        else:
            mu  = train_data.mean(axis=(0, 1), keepdims=True)  # (1, 1, n_freq)
            std = train_data.std(axis=(0, 1), keepdims=True) + 1e-8
            train_data = (train_data - mu) / std
            test_data  = (test_data  - mu) / std

        if not self.include_phase:
            train_data = np.ascontiguousarray(train_data[:, np.newaxis], dtype=np.float32)
            test_data  = np.ascontiguousarray(test_data[:,  np.newaxis], dtype=np.float32)
        else:
            train_data = np.ascontiguousarray(train_data, dtype=np.float32)
            test_data  = np.ascontiguousarray(test_data,  dtype=np.float32)

        return train_data, train_label, test_data, test_label

    def pretrain_once(self, subject_ids: list, save_dir: str, n_epochs: int = 40, batch_size: int = 128, patience: int = 10):
        print(f"\n[全局预训练] 正在加载 {len(subject_ids)} 个被试的数据并构建全局 Train / Val 集...")
        all_train_data, all_train_label = [], []
        all_val_data, all_val_label = [], []
        
        for sid in subject_ids:
            tr_d, tr_l, val_d, val_l = self._get_train_test_data(sid)
            all_train_data.append(tr_d)
            all_train_label.append(tr_l)
            all_val_data.append(val_d)
            all_val_label.append(val_l)

        all_train_data  = np.concatenate(all_train_data,  axis=0)
        all_train_label = np.concatenate(all_train_label, axis=0)
        all_val_data    = np.concatenate(all_val_data,    axis=0)
        all_val_label   = np.concatenate(all_val_label,   axis=0)

        perm = np.random.permutation(len(all_train_data))
        all_train_data  = all_train_data[perm]
        all_train_label = all_train_label[perm]

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(all_train_data,  dtype=torch.float32),
            torch.tensor(all_train_label, dtype=torch.long)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(all_val_data,  dtype=torch.float32),
            torch.tensor(all_val_label, dtype=torch.long)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=5e-4)
        scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=n_epochs)
        ema = EMA(self.model, decay=0.999)

        best_val_acc = 0.0
        patience_counter = 0
        best_save_path = os.path.join(save_dir, f'D{self.depth}_H4_S{self.emb_size}_best1.pth')

        print(f" 全局训练集大小: {len(all_train_data)} | 全局验证集大小: {len(all_val_data)}")

        for epoch in range(n_epochs):
            self.model.train()
            train_loss, train_correct = 0.0, 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                imgs = ExGAN.augment(imgs, freq_mask_width=self.freq_mask_width,
                                     freq_mask_n=self.freq_mask_n, noise_std=0.005)
                imgs, labels_a, labels_b, lam = ExGAN.mixup(imgs, labels, alpha=0.2)
                _, outputs   = self.model(imgs)
                loss = lam * self.criterion_cls(outputs, labels_a) + \
                       (1 - lam) * self.criterion_cls(outputs, labels_b)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update()

                train_loss    += loss.item() * len(imgs)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            scheduler.step()
            avg_train_loss = train_loss / len(all_train_data)
            avg_train_acc  = train_correct / len(all_train_data)

            self.model.eval()
            ema.apply_shadow()
            val_loss, val_correct = 0.0, 0

            with torch.no_grad():
                for v_imgs, v_labels in val_loader:
                    v_imgs, v_labels = v_imgs.cuda(), v_labels.cuda()
                    _, v_outputs     = self.model(v_imgs)
                    v_loss           = self.criterion_cls(v_outputs, v_labels)

                    val_loss    += v_loss.item() * len(v_imgs)
                    val_correct += (v_outputs.argmax(1) == v_labels).sum().item()

            ema.restore()
            avg_val_loss = val_loss / len(all_val_data)
            avg_val_acc  = val_correct / len(all_val_data)
            
            print(f"  Epoch {epoch+1:2d}/{n_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}")

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                ema.apply_shadow()
                torch.save(self.model.state_dict(), best_save_path)
                ema.restore()
                patience_counter = 0
                print(f"new best:save to: {best_save_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"连续 {patience} 个 Epoch 验证集精度未提升，触发 Early Stopping，提前终止预训练！")
                break

        print(f"\n[全局预训练结束] 历史最高验证集精度 (Best Val Acc): {best_val_acc * 100:.2f}%")
        return best_save_path

    def finetune_once(self, sid: int, save_dir: str, n_epochs: int = 15, batch_size: int = 32):
        train_data, train_label, test_data, test_label = self._get_train_test_data(sid)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_data,  dtype=torch.float32),
            torch.tensor(train_label, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        test_data_gpu  = torch.tensor(test_data,  dtype=torch.float32).cuda()
        test_label_gpu = torch.tensor(test_label, dtype=torch.long).cuda()

        finetune_lr = self.lr * 0.1
        optimizer   = torch.optim.Adam(self.model.parameters(), lr=finetune_lr, betas=(self.b1, self.b2))
        scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=2, total_epochs=n_epochs)
        ema = EMA(self.model, decay=0.999)

        best_acc = 0.0
        best_save_path = os.path.join(save_dir, f'HC{sid}_finetuned_best.pth')

        for epoch in range(n_epochs):
            self.model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                imgs = ExGAN.augment(imgs, freq_mask_width=self.freq_mask_width,
                                     freq_mask_n=self.freq_mask_n, noise_std=0.005)
                imgs, labels_a, labels_b, lam = ExGAN.mixup(imgs, labels, alpha=0.2)
                _, outputs   = self.model(imgs)
                loss = lam * self.criterion_cls(outputs, labels_a) + \
                       (1 - lam) * self.criterion_cls(outputs, labels_b)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update()
            scheduler.step()

            self.model.eval()
            ema.apply_shadow()
            with torch.no_grad():
                _, cls_out = self.model(test_data_gpu)
            ema.restore()

            y_pred = cls_out.argmax(dim=1)
            acc    = (y_pred == test_label_gpu).float().mean().item()

            if acc > best_acc:
                best_acc = acc
                ema.apply_shadow()
                torch.save(self.model.state_dict(), best_save_path)
                ema.restore()

        return best_acc

def warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """前 warmup_epochs 线性升温，之后余弦退火"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    DATA_DIR = "./EEG-Conformer/data/processed_normal/"
    SAVE_DIR = "./EEG-Conformer/last_params/"
    emb_size = 40
    depth   = 4

    subject_ids = sorted([
        int(os.path.basename(f).replace('HC', '').replace('_1s.mat', ''))
        for f in glob.glob(os.path.join(DATA_DIR, 'HC*_1s.mat'))
    ])
    n_subjects = len(subject_ids)
    
    if n_subjects == 0:
        print(f"未在 {DATA_DIR} 找到数据，请检查路径。")
        return
        
    print(f"找到 {n_subjects} 个受试者: {subject_ids}")

    # PSD 配置（实验 B：FFT 恢复 50 bins）
    psd_method = 'fft'
    psd_nperseg = 125
    n_freqs = ExGAN.get_n_freqs(fs=250, method=psd_method, nperseg=psd_nperseg)
    print(f"[Info] PSD method={psd_method}, 1-50Hz frequency bins = {n_freqs}")
    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=n_freqs, emb_size=16)
    print(f"[Info] 动态推理 seq_len = {seq_len}")

    starttime = datetime.datetime.now()

    global_trainer = ExGAN(data_dir=DATA_DIR, seq_len=seq_len, depth=depth,
                           emb_size=emb_size, psd_method=psd_method)
    
    best_weights_path = global_trainer.pretrain_once(
        subject_ids=subject_ids, 
        save_dir=SAVE_DIR,
        n_epochs=150,       
        batch_size=128,
        patience=30        
    )

if __name__ == "__main__":
    main()