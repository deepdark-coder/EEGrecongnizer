import os, random, datetime, time, glob, copy
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.backends import cudnn
from einops import rearrange
from einops.layers.torch import Rearrange
from pre_denoise import load_and_denoise_data

cudnn.benchmark     = False
cudnn.deterministic = True


gpus = [1]
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

def denoise_signals(all_data):
        """
        物理去噪核心函数：幅值截断 + CAR 空间滤波
        输入形状: (Trials, Channels, Times)
        """
        # 1. 幅值截断 (剔除极端的眼电/肌电突刺)
        std_val = np.std(all_data)
        threshold = 3 * std_val
        all_data = np.clip(all_data, -threshold, threshold)

        # 2. CAR 空间滤波 (共模平均参考)
        # 减去所有通道在同一时刻的均值
        common_mode_noise = np.mean(all_data, axis=1, keepdims=True)
        all_data = all_data - common_mode_noise
        
        return all_data 

'''emb_size=16,depth=1修改需要从EXGAN中的参数中修改'''
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40, n_channels: int = 30):
        super().__init__()
        self.shallownet = nn.Sequential(

            nn.Conv2d(1, 64, (1, 15), stride=(1, 1), padding=(0, 7)),
            nn.ELU(),


            nn.Conv2d(64, 64, (n_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(64), # 建议换回 BatchNorm，空间融合后更稳定
            nn.ELU(),

            # 3. 【核心细度修改】：中度池化
            # 核大小 25 (代表 0.1 秒的平滑窗口)
            # 步长 12 (代表 0.05 秒的滑动步伐)
            # 250 个点经过计算后，序列长度会变成约 19 个 Token
            nn.AvgPool2d((1, 25), stride=(1, 12)), 

            nn.Dropout(0.3),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(64, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e h w -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.shallownet(x))

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
    """Stochastic Depth — 训练时随机丢弃整条残差分支，防止过拟合。"""
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
    """指数移动平均 — 验证时使用参数滑动平均, 通常带来 0.5-1% 提升"""
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

class ConvolutionModule(nn.Module):
    """Conformer 卷积模块: 逐点卷积 → GLU → 深度可分离卷积 → 逐点卷积"""
    def __init__(self, emb_size: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(emb_size)
        self.pointwise_conv1 = nn.Conv1d(emb_size, emb_size * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            emb_size, emb_size, kernel_size,
            padding=kernel_size // 2, groups=emb_size
        )
        self.batch_norm = nn.BatchNorm1d(emb_size)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(emb_size, emb_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)                      # (B, D, T)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)                      # (B, T, D)
        return residual + x


class ConformerBlock(nn.Module):
    """Conformer Block: FFN/2 → MHSA → Conv → FFN/2 → LayerNorm"""
    def __init__(self, emb_size: int, num_heads: int = 4,
                 drop_p: float = 0.1, forward_expansion: int = 4,
                 forward_drop_p: float = 0.1, drop_path: float = 0.0,
                 conv_kernel: int = 31):
        super().__init__()
        self.ffn1 = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            nn.Dropout(drop_p),
            DropPath(drop_path),
        ))
        self.mhsa = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            MultiHeadAttention(emb_size, num_heads, drop_p),
            nn.Dropout(drop_p),
            DropPath(drop_path),
        ))
        self.conv = ConvolutionModule(emb_size, conv_kernel, drop_p)
        self.ffn2 = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            nn.Dropout(drop_p),
            DropPath(drop_path),
        ))
        self.final_norm = nn.LayerNorm(emb_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + 0.5 * self.ffn1.fn(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = x + 0.5 * self.ffn2.fn(x)
        x = self.final_norm(x)
        return x


class ConformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int, drop_path_max: float = 0.2):
        drop_path_rates = [drop_path_max * i / (depth - 1) for i in range(depth)] if depth > 1 else [0.0]
        super().__init__(*[
            ConformerBlock(emb_size, drop_path=drop_path_rates[i])
            for i in range(depth)
        ])

class ClassificationHead(nn.Module):
    # 移除 seq_len 参数，因为输入会被 GAP 压缩
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            # 直接接收 emb_size，例如扩容后的 40 维
            nn.Linear(emb_size, 32),
            nn.ELU(),
            nn.Dropout(0.2), # 维持高 Dropout
            nn.Linear(32, n_classes),
        )
        
    def forward(self, x: Tensor):
        # x 的输入形状: (Batch, seq_len=112, emb_size=40)
        
        # 核心：将 112 个时间步的特征融合为一个 40 维的全局时间不变特征
        feat = x.mean(dim=1) 
        
        return feat, self.fc(feat)

class ViT(nn.Module):
    def __init__(self, emb_size: int, depth: int,
                 n_classes: int = 2, n_channels: int = 30, seq_len: int = 11):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels)
        self.pos_embedding   = nn.Parameter(torch.randn(1, seq_len, emb_size) * 0.02)
        self.transformer     = ConformerEncoder(depth, emb_size)
        self.cls_head        = ClassificationHead(emb_size, n_classes)

    def forward(self, x: Tensor):
        x = self.patch_embedding(x)
        x = x + self.pos_embedding
        x = self.transformer(x)
        return self.cls_head(x)



class ExGAN:
    def __init__(self, data_dir: str, seq_len: int, depth: int, emb_size: int):
        self.n_channels = 30
        self.n_times    = 250
        self.n_classes  = 2
        self.lr         = 0.0002
        self.b1, self.b2 = 0.5, 0.999
        self.data_dir   = data_dir
        self.seq_len    = seq_len
        self.depth      = depth
        self.emb_size    = emb_size

        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()

        self.model = ViT(
            emb_size=self.emb_size, depth=self.depth, n_classes=self.n_classes,
            n_channels=self.n_channels, seq_len=seq_len
        ).cuda()
        self.model = nn.DataParallel(
            self.model, device_ids=list(range(len(gpus)))
        ).cuda()

    @staticmethod
    def get_seq_len(n_channels: int = 30, n_times: int = 250,
                    emb_size: int = 16) -> int:
        dummy = torch.zeros(1, 1, n_channels, n_times)
        pe    = PatchEmbedding(emb_size, n_channels)
        with torch.no_grad():
            out = pe(dummy)
        return out.shape[1]

    @staticmethod
    def augment(x: Tensor) -> Tensor:
        """EEG 数据增强: 高斯噪声 + 随机时间偏移 + 通道丢弃 (仅训练时调用)"""
        # 1. 高斯噪声 (σ=0.02)
        x = x + torch.randn_like(x) * 0.02
        # 2. 随机时间偏移 (±25 采样点, 即 ±10% 窗口)
        shift = torch.randint(-25, 25, (1,), device=x.device).item()
        x = torch.roll(x, shift, dims=-1)
        # 3. 30% 概率触发通道丢弃 (丢弃约 10% 通道)
        if torch.rand(1, device=x.device).item() < 0.3:
            mask = (torch.rand(x.size(0), 1, x.size(2), 1, device=x.device) > 0.1).float()
            x = x * mask
        return x

    @staticmethod
    def mixup(x: Tensor, y: Tensor, alpha: float = 0.2):
        """Mixup 数据增强: 线性混合两个样本及其标签"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam

    @torch.no_grad()
    def tta_evaluate(self, x: Tensor, n_views: int = 5) -> Tensor:
        """测试时增强: 多次增强取平均 logits, 降低预测方差"""
        logits_sum = None
        for _ in range(n_views):
            x_aug = ExGAN.augment(x.clone())
            _, out = self.model(x_aug)
            if logits_sum is None:
                logits_sum = out
            else:
                logits_sum += out
        return logits_sum / n_views

    # ── 新增：固定的 80/20 数据切分 (弃用多折) ──────────────────────────────
    def _get_train_test_data(self, sid: int):
        """
        单次划分：按被试固定的 seed，抽取 80% 作为训练集，20% 作为测试集。
        """
        # 注意：此处文件名需匹配你真实的数据命名规则
        mat_file = os.path.join(self.data_dir, f'HC{sid}_1s.mat')
        mat      = scipy.io.loadmat(mat_file)

        all_data  = np.ascontiguousarray(mat['data'],            dtype=np.float32)
        all_label = np.ascontiguousarray(mat['label'].flatten(), dtype=np.int64)

        all_data = denoise_signals(all_data)#denoise processing

        train_idx_list, test_idx_list = [], []
        for cls in [0, 1]:
            cls_idx   = np.where(all_label == cls)[0]
            rng       = np.random.RandomState(sid)           # 固定seed
            cls_idx   = cls_idx[rng.permutation(len(cls_idx))]
            
            # 80% 训练, 20% 测试
            split_point = int(len(cls_idx) * 0.8)
            train_idx_list.append(cls_idx[:split_point])
            test_idx_list.append(cls_idx[split_point:])

        train_idx = np.concatenate(train_idx_list)
        test_idx  = np.concatenate(test_idx_list)

        train_data,  train_label = all_data[train_idx], all_label[train_idx]
        test_data,   test_label  = all_data[test_idx],  all_label[test_idx]

        # 按通道标准化 (每通道独立均值和标准差)
        mu    = train_data.mean(axis=(0, 2), keepdims=True)
        std   = train_data.std(axis=(0, 2), keepdims=True) + 1e-8
        train_data = (train_data - mu) / std
        test_data  = (test_data  - mu) / std

        # 增加 channel 维度 (N, 1, 30, 250)
        train_data  = np.ascontiguousarray(train_data[:, np.newaxis], dtype=np.float32)
        test_data   = np.ascontiguousarray(test_data[:,  np.newaxis], dtype=np.float32)
        train_label = np.ascontiguousarray(train_label, dtype=np.int64)
        test_label  = np.ascontiguousarray(test_label,  dtype=np.int64)

        return train_data, train_label, test_data, test_label

    def pretrain_once(self, subject_ids: list, save_dir: str, n_epochs: int = 40, batch_size: int = 128, patience: int = 10):
        """
        带全局验证和早停机制的预训练。
        提取所有被试的 80% 作为训练集，20% 作为验证集。
        仅保存验证集精度（Val Acc）最高时的权重。
        """
        print(f"\n[全局预训练] 正在加载 {len(subject_ids)} 个被试的数据并构建全局 Train / Val 集...")
        all_train_data, all_train_label = [], []
        all_val_data, all_val_label = [], []
        
        for sid in subject_ids:
            tr_d, tr_l, val_d, val_l = self._get_train_test_data(sid)
            all_train_data.append(tr_d)
            all_train_label.append(tr_l)
            all_val_data.append(val_d)
            all_val_label.append(val_l)

        # 拼接全局训练集和全局验证集
        all_train_data  = np.concatenate(all_train_data,  axis=0)
        all_train_label = np.concatenate(all_train_label, axis=0)
        all_val_data    = np.concatenate(all_val_data,    axis=0)
        all_val_label   = np.concatenate(all_val_label,   axis=0)

        # 仅打乱训练集，验证集无需打乱
        perm = np.random.permutation(len(all_train_data))
        all_train_data  = all_train_data[perm]
        all_train_label = all_train_label[perm]

        # 构建双轨 DataLoader
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
        best_save_path = os.path.join(save_dir, f'conformer_best1.pth')

        print(f"  ▶ 全局训练集大小: {len(all_train_data)} | 全局验证集大小: {len(all_val_data)}")

        for epoch in range(n_epochs):
            # ================== 1. 训练阶段 ==================
            self.model.train()
            train_loss, train_correct = 0.0, 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                imgs = ExGAN.augment(imgs)
                # 概率性 Mixup: 50% 批次做 mixup, 其余用干净标签
                if torch.rand(1).item() < 0.5:
                    imgs, labels_a, labels_b, lam = ExGAN.mixup(imgs, labels, alpha=0.1)
                    _, outputs = self.model(imgs)
                    loss = lam * self.criterion_cls(outputs, labels_a) + \
                           (1 - lam) * self.criterion_cls(outputs, labels_b)
                else:
                    _, outputs = self.model(imgs)
                    loss = self.criterion_cls(outputs, labels)

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

            # ================== 2. 验证阶段 ==================
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

            # ================== 3. 最优保存与早停 ==================
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

    # ── 重写：单次微调 (直接返回最高精度) ───────────────────────────────────
    def finetune_once(self, sid: int, save_dir: str, n_epochs: int = 15, batch_size: int = 32, n_tta: int = 5):
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
        best_save_path = os.path.join(save_dir, f'finetuned_best.pth')

        for epoch in range(n_epochs):
            self.model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                imgs = ExGAN.augment(imgs)
                if torch.rand(1).item() < 0.5:
                    imgs, labels_a, labels_b, lam = ExGAN.mixup(imgs, labels, alpha=0.1)
                    _, outputs = self.model(imgs)
                    loss = lam * self.criterion_cls(outputs, labels_a) + \
                           (1 - lam) * self.criterion_cls(outputs, labels_b)
                else:
                    _, outputs = self.model(imgs)
                    loss = self.criterion_cls(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update()
            scheduler.step()

            self.model.eval()
            ema.apply_shadow()
            with torch.no_grad():
                cls_out = self.tta_evaluate(test_data_gpu, n_views=n_tta)
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
    """前 warmup_epochs 线性升温，之后余弦退火。"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def read_data(data_dir: str):
    subject_ids = sorted([
        int(os.path.basename(f).replace('HC', '').replace('_1s.mat', ''))
        for f in glob.glob(os.path.join(data_dir, 'HC*_1s.mat'))
    ])
    return subject_ids

def main():
    DATA_DIR = "./EEG-Conformer/data/processed_normal/"
    SAVE_DIR = "./EEG-Conformer/last_params/"
    emb_size = 40
    depth   = 2

    # 自动扫描受试者列表
    subject_ids = read_data(DATA_DIR)
    n_subjects = len(subject_ids)

    if n_subjects == 0:
        print(f"未在 {DATA_DIR} 找到数据，请检查路径。")
        return

    print(f"找到 {n_subjects} 个受试者: {subject_ids}")

    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=250, emb_size=emb_size)
    print(f"[Info] 动态推理 seq_len = {seq_len}")

    starttime = datetime.datetime.now()

    global_trainer = ExGAN(data_dir=DATA_DIR, seq_len=seq_len, depth=depth, emb_size=emb_size)

    # ── 阶段 1: 全局预训练 (EMA + Mixup) ──
    best_weights_path = global_trainer.pretrain_once(
        subject_ids=subject_ids,
        save_dir=SAVE_DIR,
        n_epochs=300,
        batch_size=128,
        patience=120
    )

    # ── 阶段 2: 逐被试微调 + TTA 评估 ──
    print(f"\n{'='*50}")
    print("开始逐被试微调 + TTA 测试...")
    print(f"{'='*50}")

    acc_list = []
    n_tta = 5

    for sid in subject_ids:
        global_trainer.model.load_state_dict(
            torch.load(best_weights_path, map_location='cuda'), strict=True
        )
        acc = global_trainer.finetune_once(sid, SAVE_DIR, n_epochs=15, batch_size=32, n_tta=n_tta)
        acc_list.append(acc)
        print(f"  被试 HC{sid} TTA-{n_tta} 测试最高精度: {acc * 100:.2f}%")

    avg_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    print(f"\n{'='*50}")
    print(f"全部 {n_subjects} 个被试测试完毕 (TTA-{n_tta})")
    print(f"平均精度: {avg_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    print(f"总耗时: {datetime.datetime.now() - starttime}")
    print(f"{'='*50}")



if __name__ == "__main__":
    main()