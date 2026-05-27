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

cudnn.benchmark     = False
cudnn.deterministic = True


gpus = [1]
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

def denoise_signals(all_data):
    std_val = np.std(all_data)
    threshold = 3 * std_val
    all_data = np.clip(all_data, -threshold, threshold)
    common_mode_noise = np.mean(all_data, axis=1, keepdims=True)
    all_data = all_data - common_mode_noise
    return all_data


# ======================== 模型组件 ========================

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40, n_channels: int = 30):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 64, (1, 15), stride=(1, 1), padding=(0, 7)),
            nn.ELU(),
            nn.Conv2d(64, 64, (n_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
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
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x


class ConformerBlock(nn.Module):
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
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 32),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: Tensor):
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


# ======================== 训练主类 ========================

class ExGAN:
    def __init__(self, data_dirs: list, seq_len: int, depth: int, emb_size: int):
        self.n_channels = 30
        self.n_times    = 250
        self.n_classes  = 2
        self.lr         = 0.0002
        self.b1, self.b2 = 0.5, 0.999
        self.data_dirs  = data_dirs
        self.seq_len    = seq_len
        self.depth      = depth
        self.emb_size   = emb_size

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
        x = x + torch.randn_like(x) * 0.02
        shift = torch.randint(-25, 25, (1,), device=x.device).item()
        x = torch.roll(x, shift, dims=-1)
        if torch.rand(1, device=x.device).item() < 0.3:
            mask = (torch.rand(x.size(0), 1, x.size(2), 1, device=x.device) > 0.1).float()
            x = x * mask
        return x

    @staticmethod
    def mixup(x: Tensor, y: Tensor, alpha: float = 0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam

    @torch.no_grad()
    def tta_evaluate(self, x: Tensor, n_views: int = 5) -> Tensor:
        logits_sum = None
        for _ in range(n_views):
            x_aug = ExGAN.augment(x.clone())
            _, out = self.model(x_aug)
            if logits_sum is None:
                logits_sum = out
            else:
                logits_sum += out
        return logits_sum / n_views

    def load_pretrained(self, weights_path: str):
        """加载预训练权重 (自动适配 DataParallel 的 module. 前缀)"""
        checkpoint = torch.load(weights_path, map_location='cuda')
        state_dict = checkpoint.get('model_state', checkpoint)

        # 探测权重文件与当前模型的 key 前缀是否匹配
        sample_key = next(iter(state_dict.keys()))
        model_sample = next(iter(self.model.state_dict().keys()))

        new_state_dict = {}
        for k, v in state_dict.items():
            if model_sample.startswith('module.') and not sample_key.startswith('module.'):
                # 模型有 module. 前缀，权重没有 → 添加
                new_state_dict[f'module.{k}'] = v
            elif not model_sample.startswith('module.') and sample_key.startswith('module.'):
                # 权重有 module. 前缀，模型没有 → 去除
                new_state_dict[k.replace('module.', '', 1)] = v
            else:
                # 前缀一致，直接复制
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=True)
        print(f"[Info] 预训练权重加载成功: {weights_path}")

    # ── 数据加载 ————————————————————————————————————————————
    def _get_train_test_data(self, data_dir: str, prefix: str, sid: int):
        mat_file = os.path.join(data_dir, f'{prefix}{sid}_1s.mat')
        mat      = scipy.io.loadmat(mat_file)

        all_data  = np.ascontiguousarray(mat['data'],            dtype=np.float32)
        all_label = np.ascontiguousarray(mat['label'].flatten(), dtype=np.int64)

        all_data = denoise_signals(all_data)

        train_idx_list, test_idx_list = [], []
        for cls in [0, 1]:
            cls_idx = np.where(all_label == cls)[0]
            rng     = np.random.RandomState(sid)
            cls_idx = cls_idx[rng.permutation(len(cls_idx))]
            split_point = int(len(cls_idx) * 0.8)
            train_idx_list.append(cls_idx[:split_point])
            test_idx_list.append(cls_idx[split_point:])

        train_idx = np.concatenate(train_idx_list)
        test_idx  = np.concatenate(test_idx_list)

        train_data, train_label = all_data[train_idx], all_label[train_idx]
        test_data,  test_label  = all_data[test_idx],  all_label[test_idx]

        mu  = train_data.mean(axis=(0, 2), keepdims=True)
        std = train_data.std(axis=(0, 2), keepdims=True) + 1e-8
        train_data = (train_data - mu) / std
        test_data  = (test_data  - mu) / std

        train_data  = np.ascontiguousarray(train_data[:, np.newaxis], dtype=np.float32)
        test_data   = np.ascontiguousarray(test_data[:,  np.newaxis], dtype=np.float32)
        train_label = np.ascontiguousarray(train_label, dtype=np.int64)
        test_label  = np.ascontiguousarray(test_label,  dtype=np.int64)

        return train_data, train_label, test_data, test_label

    @staticmethod
    def read_subject_list(data_dirs: list):
        subject_list = []
        for data_dir, prefix in data_dirs:
            pattern = os.path.join(data_dir, f'{prefix}*_1s.mat')
            for f in glob.glob(pattern):
                fname = os.path.basename(f)
                sid = int(fname.replace(prefix, '').replace('_1s.mat', ''))
                subject_list.append((data_dir, prefix, sid))
        subject_list.sort(key=lambda x: (x[1], x[2]))
        return subject_list

    # ── 后训练：在预训练权重基础上，用混合数据继续训练 —————————
    def posttrain_once(self, subject_list: list, save_dir: str,
                       n_epochs: int = 100, batch_size: int = 128, patience: int = 20,
                       post_lr: float = None):
        """
        基于已加载的预训练权重，使用混合数据集进行后训练。
        post_lr: 后训练学习率，默认使用 self.lr * 0.1 (更小的学习率避免灾难性遗忘)
        """
        if post_lr is None:
            post_lr = self.lr * 0.1

        print(f"\n[后训练 - MixPosttrain] 学习率: {post_lr:.6f} | 被试数: {len(subject_list)}")

        all_train_data, all_train_label = [], []
        all_val_data,   all_val_label   = [], []

        for data_dir, prefix, sid in subject_list:
            tr_d, tr_l, val_d, val_l = self._get_train_test_data(data_dir, prefix, sid)
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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=post_lr, betas=(self.b1, self.b2), weight_decay=5e-4)
        scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=3, total_epochs=n_epochs)
        ema = EMA(self.model, decay=0.999)

        best_val_acc = 0.0
        patience_counter = 0
        best_save_path = os.path.join(save_dir, f'mixpretrain_D{self.depth}_H4_S{self.emb_size}_best.pth')

        print(f"  ▶ 全局训练集: {len(all_train_data)} | 全局验证集: {len(all_val_data)}")

        for epoch in range(n_epochs):
            self.model.train()
            train_loss, train_correct = 0.0, 0

            for imgs, labels in train_loader:
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
                print(f"  [new best] saved → {best_save_path}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  连续 {patience} epoch 未提升，Early Stopping。")
                break

        print(f"\n[后训练结束] Best Val Acc: {best_val_acc * 100:.2f}%")
        return best_save_path

    # ── 逐被试微调 ———————————————————————————————————————————
    def finetune_once(self, data_dir: str, prefix: str, sid: int, save_dir: str,
                      n_epochs: int = 15, batch_size: int = 32, n_tta: int = 5):
        train_data, train_label, test_data, test_label = \
            self._get_train_test_data(data_dir, prefix, sid)

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
        best_save_path = os.path.join(save_dir, f'{prefix}_mixpretrain_finetuned_best.pth')

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
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    # ── 配置 ────────────────────────────────────────────────
    # 预训练权重 (仅在 Normal 数据上训练的)
    PRETRAIN_WEIGHTS = "./EEG-Conformer/last_params/D2_H4_S40_best1.pth"

    # 混合数据源: Normal + Depressive
    DATA_DIRS = [
        ('./EEG-Conformer/data/processed_normal/',     'HC'),
        ('./EEG-Conformer/data/processed_depressive/', 'DEP'),
    ]
    SAVE_DIR  = "./EEG-Conformer/last_params/"
    emb_size  = 40
    depth     = 2

    # ── 扫描被试 ————————————————————————————————————————————
    subject_list = ExGAN.read_subject_list(DATA_DIRS)
    print(f"扫描到 {len(subject_list)} 个被试:")
    for d, p, sid in subject_list:
        print(f"  {p}{sid}  ({d})")

    if len(subject_list) == 0:
        print("未找到任何数据，请检查目录。")
        return

    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=250, emb_size=emb_size)
    print(f"[Info] seq_len = {seq_len}")

    starttime = datetime.datetime.now()

    trainer = ExGAN(data_dirs=DATA_DIRS, seq_len=seq_len, depth=depth, emb_size=emb_size)

    # ── 阶段 0: 加载 Normal 预训练权重 ─——————————————————————
    trainer.load_pretrained(PRETRAIN_WEIGHTS)

    # ── 阶段 1: 在混合数据上后训练 ─———————————————————————————
    best_weights_path = trainer.posttrain_once(
        subject_list=subject_list,
        save_dir=SAVE_DIR,
        n_epochs=200,
        batch_size=128,
        patience=40,
    )

    # ── 阶段 2: 逐被试微调 + TTA 评估 ————————————————————————
    print(f"\n{'='*50}")
    print("逐被试微调 + TTA 测试...")
    print(f"{'='*50}")

    acc_list = []
    n_tta = 5

    for data_dir, prefix, sid in subject_list:
        trainer.model.load_state_dict(
            torch.load(best_weights_path, map_location='cuda'), strict=True
        )
        acc = trainer.finetune_once(data_dir, prefix, sid, SAVE_DIR,
                                    n_epochs=15, batch_size=32, n_tta=n_tta)
        acc_list.append(acc)
        print(f"  被试 {prefix}{sid} TTA-{n_tta} 最高精度: {acc * 100:.2f}%")

    avg_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    print(f"\n{'='*50}")
    print(f"全部 {len(subject_list)} 个被试测试完毕 (TTA-{n_tta})")
    print(f"平均精度: {avg_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    print(f"总耗时: {datetime.datetime.now() - starttime}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
