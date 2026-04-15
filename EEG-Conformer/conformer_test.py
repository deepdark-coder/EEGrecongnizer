"""
EEG Conformer — SEED dataset, 1-second epochs, strict 5-fold cross-validation
Data shape: (400, 30, 250),  Label shape: (400,)
"""

import os
import math
import random
import datetime
import time
import scipy.io
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.backends import cudnn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

cudnn.benchmark = False
cudnn.deterministic = True

# ─────────────────────────── GPU 设置（按需修改） ────────────────────────────
gpus = [0]                          # ← 修改为你实际使用的GPU编号
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))


# ══════════════════════════════ 模型定义 ══════════════════════════════════════

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40, n_channels: int = 30):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.Conv2d(40, 40, (n_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), stride=(1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e h w -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.emb_size  = emb_size
        self.num_heads = num_heads
        self.keys      = nn.Linear(emb_size, emb_size)
        self.queries   = nn.Linear(emb_size, emb_size)
        self.values    = nn.Linear(emb_size, emb_size)
        self.att_drop  = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys    = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)
        energy  = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            energy = energy.masked_fill_(~mask, torch.finfo(torch.float32).min)  # FIX: masked_fill_
        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


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
    def __init__(self, emb_size: int, num_heads: int = 8,
                 drop_p: float = 0.5, forward_expansion: int = 4,
                 forward_drop_p: float = 0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int, n_classes: int, seq_len: int = 7):
        """
        seq_len: PatchEmbedding 输出的序列长度，由输入时间长度决定
                 input (1,30,250) → shallownet → (40,1,12) → projection → (12, emb_size)
                 所以 seq_len=12，flatten_dim = 12 * emb_size = 12*40 = 480
                 根据实际输出动态计算，见 ViT.forward 里的 print 调试
        """
        super().__init__()
        flatten_dim = seq_len * emb_size          # ← 动态传入，不再硬编码280
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),             # FIX: 使用 n_classes 而非硬编码3
        )

    def forward(self, x: Tensor):
        feat = x.contiguous().view(x.size(0), -1)
        out  = self.fc(feat)
        return feat, out


class ViT(nn.Module):
    def __init__(self, emb_size: int = 40, depth: int = 6,
                 n_classes: int = 3, n_channels: int = 30, seq_len: int = 12):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels)
        self.transformer     = TransformerEncoder(depth, emb_size)
        self.cls_head        = ClassificationHead(emb_size, n_classes, seq_len)

    def forward(self, x: Tensor):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        return self.cls_head(x)


# ══════════════════════════════ 训练器 ═════════════════════════════════════════

class ExGAN:                                      # FIX: 去掉错误的 super().__init__()
    def __init__(self, nsub: int, fold: int, data_dir: str, log_dir: str, seq_len: 12):
        self.batch_size = 64
        self.n_epochs   = 40
        self.n_channels = 30
        self.n_times    = 250
        self.n_classes  = 2                       # SEED: negative/neutral/positive
        self.lr         = 0.0002
        self.b1, self.b2 = 0.5, 0.999
        self.nSub       = nsub

        # ── 路径 ──────────────────────────────────────────────────────────────
        self.data_dir = data_dir                  # ← 从外部传入，不再硬编码
        os.makedirs(log_dir, exist_ok=True)
        self.log_write = open(
            os.path.join(log_dir, f"log_subject{nsub:02d}_fold{fold+1}.txt"), "w"
        )

        self.Tensor     = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        # seq_len = 12 for input (1, 30, 250); verify with get_seq_len() if unsure
        self.model = ViT(emb_size=40, depth=2, n_classes=self.n_classes,
                         n_channels=self.n_channels, seq_len=seq_len).cuda()
        self.model = nn.DataParallel(self.model, device_ids=list(range(len(gpus)))).cuda()

    @staticmethod
    def get_seq_len(n_channels: int = 30, n_times: int = 250, emb_size: int = 40) -> int:
        """
        用假数据推算 PatchEmbedding 输出的序列长度，避免手动计算出错。
        """
        dummy = torch.zeros(1, 1, n_channels, n_times)
        pe    = PatchEmbedding(emb_size, n_channels)
        with torch.no_grad():
            out = pe(dummy)          # (1, seq_len, emb_size)
        return out.shape[1]

    # ── 数据加载（完全重写） ──────────────────────────────────────────────────
    def get_source_data(self, fold: int):
        mat_file = os.path.join(self.data_dir, f'HC{self.nSub}_1s.mat')
        mat      = scipy.io.loadmat(mat_file)

        all_data  = np.ascontiguousarray(mat['data'],           dtype=np.float32)
        all_label = np.ascontiguousarray(mat['label'].flatten(), dtype=np.int64)

        # ── 分层5折：每类内部用固定seed划分，保证：
        #    1. 每折test集类别均衡（各40个）
        #    2. 同一subject不同fold之间test不重叠
        #    3. 可复现
        train_idx_list, test_idx_list = [], []
        for cls in [0, 1]:
            cls_idx = np.where(all_label == cls)[0]          # 各200个
            # 用 subject_id 作为seed，保证同一subject每次运行结果一致
            rng     = np.random.RandomState(self.nSub)
            cls_idx = cls_idx[rng.permutation(len(cls_idx))]

            fold_size = len(cls_idx) // 5                    # 40
            t_s, t_e  = fold * fold_size, (fold + 1) * fold_size
            test_idx_list .append(cls_idx[t_s:t_e])          # 40个
            train_idx_list.append(np.concatenate([cls_idx[:t_s], cls_idx[t_e:]]))  # 160个

        train_idx = np.concatenate(train_idx_list)           # 320
        test_idx  = np.concatenate(test_idx_list)            # 80

        train_data,  train_label = all_data[train_idx], all_label[train_idx]
        test_data,   test_label  = all_data[test_idx],  all_label[test_idx]

        # 训练集打乱（每次调用都重新打乱，DataLoader也会打乱，双重保障）
        perm       = np.random.permutation(len(train_data))
        train_data = train_data[perm]
        train_label= train_label[perm]

        # 标准化
        mu, std    = train_data.mean(), train_data.std() + 1e-8
        train_data = (train_data - mu) / std
        test_data  = (test_data  - mu) / std

        # 增加channel维度
        train_data  = np.ascontiguousarray(train_data[:, np.newaxis], dtype=np.float32)
        test_data   = np.ascontiguousarray(test_data[:,  np.newaxis], dtype=np.float32)
        train_label = np.ascontiguousarray(train_label, dtype=np.int64)
        test_label  = np.ascontiguousarray(test_label,  dtype=np.int64)

        print(f"    Train: {dict(zip(*np.unique(train_label, return_counts=True)))}")
        print(f"    Test:  {dict(zip(*np.unique(test_label,  return_counts=True)))}")
        return train_data, train_label, test_data, test_label

    # ── 训练主循环 ────────────────────────────────────────────────────────────
    def train(self, fold: int):
        train_data, train_label, test_data, test_label = self.get_source_data(fold)

        # FIX: label 已是 0/1/2，不再 +1
        train_tensor  = torch.tensor(train_data,  dtype=torch.float32)
        label_tensor  = torch.tensor(train_label, dtype=torch.long)
        test_tensor   = torch.tensor(test_data,   dtype=torch.float32)
        tlabel_tensor = torch.tensor(test_label,  dtype=torch.long)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_tensor, label_tensor),
            batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr, betas=(self.b1, self.b2))

        test_data_gpu  = Variable(test_tensor.type(self.Tensor))
        test_label_gpu = Variable(tlabel_tensor.type(self.LongTensor))

        best_acc, aver_acc, n_eval = 0.0, 0.0, 0
        Y_true, Y_pred = None, None

        for epoch in range(self.n_epochs):
            self.model.train()
            for imgs, labels in loader:
                imgs   = Variable(imgs.type(self.Tensor))
                labels = Variable(labels.type(self.LongTensor))

                _, outputs = self.model(imgs)
                loss = self.criterion_cls(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ── 每个 epoch 评估 ───────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                _, cls_out = self.model(test_data_gpu)

            loss_test  = self.criterion_cls(cls_out, test_label_gpu)
            y_pred     = cls_out.argmax(dim=1)
            acc        = (y_pred == test_label_gpu).float().mean().item()

            train_pred = outputs.argmax(dim=1)
            train_acc  = (train_pred == labels).float().mean().item()

            print(f"Epoch {epoch:3d} | "
                  f"Train loss: {loss.item():.4f}  Train acc: {train_acc:.4f} | "
                  f"Test  loss: {loss_test.item():.4f}  Test  acc: {acc:.4f}")
            self.log_write.write(f"{epoch}\t{acc:.6f}\n")

            n_eval   += 1
            aver_acc += acc
            if acc > best_acc:
                best_acc = acc
                Y_true   = test_label_gpu
                Y_pred   = y_pred

        aver_acc /= n_eval
        print(f"[Fold {fold+1}] Best: {best_acc:.4f}  Aver: {aver_acc:.4f}")
        self.log_write.write(f"Best: {best_acc}\nAver: {aver_acc}\n")
        self.log_write.close()
        return best_acc, aver_acc, Y_true, Y_pred


# ══════════════════════════════ 主函数 ════════════════════════════════════════

def main():
    # !! 修改这两个路径 !!
    DATA_DIR = "./EEG-Conformer/data/processed_normal/"           # ← 存放 HC0001_1s.mat 等文件的目录
    LOG_DIR  = "./EEG-Conformer/params/"            # ← 日志和结果的输出目录

    os.makedirs(LOG_DIR, exist_ok=True)
        # 自动扫描受试者列表
    subject_ids = sorted([
        int(os.path.basename(f).replace('HC', '').replace('_1s.mat', ''))
        for f in glob.glob(os.path.join(DATA_DIR, 'HC*_1s.mat'))
    ])
    n_subjects = len(subject_ids)
    print(f"找到 {n_subjects} 个受试者: {subject_ids}")

    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=250, emb_size=40)
    print(f"[Info] PatchEmbedding output seq_len = {seq_len}")

    best_all, aver_all = 0.0, 0.0

    with open(os.path.join(LOG_DIR, "sub_result.txt"), "w") as result_write:
        for i, sub_idx in enumerate(subject_ids):
            starttime = datetime.datetime.now()
            seed_n    = np.random.randint(2021)

            random.seed(seed_n)
            np.random.seed(seed_n)
            torch.manual_seed(seed_n)
            torch.cuda.manual_seed_all(seed_n)

            print(f"\n{'='*50}\nSubject HC{sub_idx}  ({i+1}/{n_subjects})  seed={seed_n}")
            result_write.write(f"Subject HC{sub_idx}  seed={seed_n}\n")

            best5, aver5 = 0.0, 0.0
            for fold in range(5):
                exgan = ExGAN(sub_idx, fold, DATA_DIR, LOG_DIR, seq_len)
                ba, aa, _, _ = exgan.train(fold)
                result_write.write(f"  Fold {fold+1}: best={ba:.4f}  aver={aa:.4f}\n")
                best5 += ba
                aver5 += aa

            best5 /= 5
            aver5 /= 5
            result_write.write(f"  5-fold best={best5:.4f}  aver={aver5:.4f}\n")
            result_write.write("-" * 50 + "\n")
            result_write.flush()

            best_all += best5
            aver_all += aver5
            print(f"HC{sub_idx} 耗时: {datetime.datetime.now() - starttime}")

        best_all /= n_subjects
        aver_all /= n_subjects
        result_write.write(f"\nAll subjects  best={best_all:.4f}  aver={aver_all:.4f}\n")
        print(f"\nAll subjects  best={best_all:.4f}  aver={aver_all:.4f}")


if __name__ == "__main__":
    print(time.asctime())
    main()
    print(time.asctime())