"""
EEG Conformer
Data shape: (400, 30, 250),  Label shape: (400,)  — 0=neutral, 1=positive
Training:   Leave-One-Subject-Out pretrain  →  per-subject 5-fold finetune
"""

import os, random, datetime, time, glob
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

# ─────────────────────────── GPU 设置 ───────────────────────────────────────
gpus = [0]
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))


# ════════════════════════════ 模型定义 ═══════════════════════════════════════

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
        att = self.att_drop(F.softmax(energy / self.emb_size ** 0.5, dim=-1))
        out = torch.einsum('bhal, bhlv -> bhav', att, v)
        return self.projection(rearrange(out, "b h n d -> b n (h d)"))


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
                FeedForwardBlock(emb_size, expansion=forward_expansion,
                                 drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int, n_classes: int, seq_len: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(seq_len * emb_size, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: Tensor):
        feat = x.contiguous().view(x.size(0), -1)
        return feat, self.fc(feat)


class ViT(nn.Module):
    def __init__(self, emb_size: int = 40, depth: int = 2,
                 n_classes: int = 2, n_channels: int = 30, seq_len: int = 11):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels)
        self.transformer     = TransformerEncoder(depth, emb_size)
        self.cls_head        = ClassificationHead(emb_size, n_classes, seq_len)

    def forward(self, x: Tensor):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        return self.cls_head(x)


# ════════════════════════════ 训练器 ══════════════════════════════════════════

class ExGAN:
    def __init__(self, nsub: int, data_dir: str, log_dir: str, seq_len: int):
        self.n_channels = 30
        self.n_times    = 250
        self.n_classes  = 2
        self.lr         = 0.0002
        self.b1, self.b2 = 0.5, 0.999
        self.nSub       = nsub
        self.data_dir   = data_dir
        self.log_dir    = log_dir
        self.seq_len    = seq_len

        os.makedirs(log_dir, exist_ok=True)

        self.criterion_cls = nn.CrossEntropyLoss().cuda()

        self.model = ViT(
            emb_size=40, depth=2, n_classes=self.n_classes,
            n_channels=self.n_channels, seq_len=seq_len
        ).cuda()
        self.model = nn.DataParallel(
            self.model, device_ids=list(range(len(gpus)))
        ).cuda()

    # ── 推算 seq_len ─────────────────────────────────────────────────────────
    @staticmethod
    def get_seq_len(n_channels: int = 30, n_times: int = 250,
                    emb_size: int = 40) -> int:
        dummy = torch.zeros(1, 1, n_channels, n_times)
        pe    = PatchEmbedding(emb_size, n_channels)
        with torch.no_grad():
            out = pe(dummy)
        return out.shape[1]

    # ── 读取单个被试全量数据（预训练用）─────────────────────────────────────
    def _load_subject(self, sid: int):
        """
        返回:
          data  (400, 1, 30, 250)  float32  — 已做 per-subject 标准化
          label (400,)             int64
        """
        mat_file = os.path.join(self.data_dir, f'HC{sid}_1s.mat')
        mat      = scipy.io.loadmat(mat_file)

        data  = np.ascontiguousarray(mat['data'],            dtype=np.float32)
        label = np.ascontiguousarray(mat['label'].flatten(), dtype=np.int64)

        # per-subject 标准化，消除被试间幅值差异
        data = (data - data.mean()) / (data.std() + 1e-8)

        # 增加 channel 维度，固化 dtype
        data = np.ascontiguousarray(data[:, np.newaxis, :, :], dtype=np.float32)
        return data, label

    # ── 读取单个被试并做分层5折划分（微调用）────────────────────────────────
    def _get_fold_data(self, sid: int, fold: int):
        """
        分层5折划分：
          每类200个样本，类内用固定 seed=sid 打乱后按 fold 切分
          test:  每类40个，共80个，类别均衡
          train: 每类160个，共320个
          用训练集统计量对 train/test 统一标准化（防数据泄露）
        """
        mat_file = os.path.join(self.data_dir, f'HC{sid}_1s.mat')
        mat      = scipy.io.loadmat(mat_file)

        all_data  = np.ascontiguousarray(mat['data'],            dtype=np.float32)
        all_label = np.ascontiguousarray(mat['label'].flatten(), dtype=np.int64)

        train_idx_list, test_idx_list = [], []
        for cls in [0, 1]:
            cls_idx   = np.where(all_label == cls)[0]
            rng       = np.random.RandomState(sid)           # 固定seed，可复现
            cls_idx   = cls_idx[rng.permutation(len(cls_idx))]
            fold_size = len(cls_idx) // 5                    # 40
            t_s, t_e  = fold * fold_size, (fold + 1) * fold_size
            test_idx_list .append(cls_idx[t_s:t_e])
            train_idx_list.append(
                np.concatenate([cls_idx[:t_s], cls_idx[t_e:]])
            )

        train_idx = np.concatenate(train_idx_list)           # 320
        test_idx  = np.concatenate(test_idx_list)            # 80

        train_data,  train_label = all_data[train_idx], all_label[train_idx]
        test_data,   test_label  = all_data[test_idx],  all_label[test_idx]

        # 打乱训练集
        perm        = np.random.permutation(len(train_data))
        train_data  = train_data[perm]
        train_label = train_label[perm]

        # 用训练集统计量标准化
        mu, std    = train_data.mean(), train_data.std() + 1e-8
        train_data = (train_data - mu) / std
        test_data  = (test_data  - mu) / std

        # 增加 channel 维度，固化 dtype
        train_data  = np.ascontiguousarray(train_data[:, np.newaxis], dtype=np.float32)
        test_data   = np.ascontiguousarray(test_data[:,  np.newaxis], dtype=np.float32)
        train_label = np.ascontiguousarray(train_label, dtype=np.int64)
        test_label  = np.ascontiguousarray(test_label,  dtype=np.int64)

        print(f"    [Fold {fold+1}] "
              f"Train dist: {dict(zip(*np.unique(train_label, return_counts=True)))} | "
              f"Test  dist: {dict(zip(*np.unique(test_label,  return_counts=True)))}")
        return train_data, train_label, test_data, test_label

    # ── 阶段一：跨被试预训练 ─────────────────────────────────────────────────
    def pretrain(self, pretrain_subject_ids: list, save_path: str,
                 n_epochs: int = 30, batch_size: int = 128):
        """
        用除目标被试外的所有被试数据联合训练通用特征提取器。
        覆盖式保存：全程只保留 loss 最低的一个权重文件。
        """
        print(f"\n  [预训练] 使用 {len(pretrain_subject_ids)} 个被试数据...")

        data_list, label_list = [], []
        for sid in pretrain_subject_ids:
            d, l = self._load_subject(sid)
            data_list .append(d)
            label_list.append(l)

        # 合并后强制固化 dtype，解决 torch 无法推断 numpy.float32 的问题
        all_data  = np.ascontiguousarray(
            np.concatenate(data_list,  axis=0), dtype=np.float32)
        all_label = np.ascontiguousarray(
            np.concatenate(label_list, axis=0), dtype=np.int64)

        # 打乱
        perm      = np.random.permutation(len(all_data))
        all_data  = np.ascontiguousarray(all_data[perm],  dtype=np.float32)
        all_label = np.ascontiguousarray(all_label[perm], dtype=np.int64)

        # 显式指定 dtype，不依赖自动推断
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(all_data,  dtype=torch.float32),
            torch.tensor(all_label, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        best_loss = float('inf')
        print(f"    总样本数: {len(all_data)}  "
              f"batch_size: {batch_size}  epochs: {n_epochs}")

        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss, n_correct, n_total = 0.0, 0, 0

            for imgs, labels in loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                _, outputs   = self.model(imgs)
                loss         = self.criterion_cls(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(imgs)
                n_correct  += (outputs.argmax(1) == labels).sum().item()
                n_total    += len(imgs)

            scheduler.step()
            avg_loss = epoch_loss / n_total
            avg_acc  = n_correct  / n_total
            print(f"    Pretrain Epoch {epoch:3d} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f} | "
                  f"loss={avg_loss:.4f}  acc={avg_acc:.4f}")

            # 覆盖式保存：只保留 loss 最低的权重
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"    ✓ 预训练权重已覆盖保存 (loss={best_loss:.4f})")

        print(f"  [预训练完成] 最优 loss={best_loss:.4f}  权重: {save_path}")

    # ── 加载预训练权重 ────────────────────────────────────────────────────────
    def load_pretrain(self, load_path: str):
        state = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(state)
        print(f"    已加载预训练权重: {load_path}")

    # ── 阶段二：目标被试微调 ─────────────────────────────────────────────────
    def finetune(self, fold: int, n_epochs: int = 60, batch_size: int = 64):
        """
        在目标被试第 fold 折数据上微调。
        覆盖式保存：每折只产生一个 .pth 文件，有提升才覆盖。
        """
        train_data, train_label, test_data, test_label = \
            self._get_fold_data(self.nSub, fold)

        # 显式指定 dtype
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_data,  dtype=torch.float32),
            torch.tensor(train_label, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        test_data_gpu  = torch.tensor(test_data,  dtype=torch.float32).cuda()
        test_label_gpu = torch.tensor(test_label, dtype=torch.long).cuda()

        # 微调用更小的 lr，避免破坏预训练特征
        finetune_lr = self.lr * 0.1
        optimizer   = torch.optim.Adam(
            self.model.parameters(), lr=finetune_lr, betas=(self.b1, self.b2)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        # 每折只产生一个覆盖式权重文件
        save_path = os.path.join(
            self.log_dir, f'HC{self.nSub}_fold{fold+1}_best.pth'
        )
        log_path = os.path.join(
            self.log_dir, f'log_HC{self.nSub}_fold{fold+1}.txt'
        )
        log_file = open(log_path, 'w')
        log_file.write("epoch\ttrain_loss\ttrain_acc\ttest_loss\ttest_acc\n")

        best_acc = 0.0
        aver_acc = 0.0
        n_eval   = 0
        Y_true   = None
        Y_pred   = None

        for epoch in range(n_epochs):
            # ── 训练 ──────────────────────────────────────────────────────
            self.model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                _, outputs   = self.model(imgs)
                loss         = self.criterion_cls(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            # ── 评估 ──────────────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                _, cls_out = self.model(test_data_gpu)

            loss_test  = self.criterion_cls(cls_out, test_label_gpu)
            y_pred     = cls_out.argmax(dim=1)
            acc        = (y_pred == test_label_gpu).float().mean().item()
            train_pred = outputs.argmax(dim=1)
            train_acc  = (train_pred == labels).float().mean().item()

            print(f"  Epoch {epoch:3d} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f} | "
                  f"Train loss:{loss.item():.4f} acc:{train_acc:.4f} | "
                  f"Test  loss:{loss_test.item():.4f} acc:{acc:.4f}")
            log_file.write(
                f"{epoch}\t{loss.item():.6f}\t{train_acc:.6f}\t"
                f"{loss_test.item():.6f}\t{acc:.6f}\n"
            )
            log_file.flush()

            n_eval   += 1
            aver_acc += acc

            # 有提升才覆盖保存，整个 fold 只保留最优的一个文件
            if acc > best_acc:
                best_acc       = acc
                Y_true, Y_pred = test_label_gpu, y_pred
                torch.save({
                    'epoch'      : epoch,
                    'acc'        : best_acc,
                    'model_state': self.model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                }, save_path)
                print(f"  ✓ [Fold {fold+1}] 最优模型已覆盖保存  "
                      f"epoch={epoch}  acc={best_acc:.4f}")

        aver_acc /= n_eval
        log_file.write(f"Best:{best_acc:.6f}\nAver:{aver_acc:.6f}\n")
        log_file.close()
        print(f"  [Fold {fold+1}] Best={best_acc:.4f}  Aver={aver_acc:.4f}")
        return best_acc, aver_acc, Y_true, Y_pred


# ════════════════════════════ 主函数 ══════════════════════════════════════════

def main():
    DATA_DIR = "./EEG-Conformer/data/processed_normal/"
    LOG_DIR  = "./EEG-Conformer/params/"
    os.makedirs(LOG_DIR, exist_ok=True)

    # 预训练权重：所有被试共用一个文件，每次覆盖
    PRETRAIN_PATH = os.path.join(LOG_DIR, "pretrain_best.pth")

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

            print(f"\n{'='*55}")
            print(f"Subject HC{sub_idx}  ({i+1}/{n_subjects})  seed={seed_n}")
            print(f"{'='*55}")
            result_write.write(f"Subject HC{sub_idx}  seed={seed_n}\n")

            # ── 阶段一：用除本被试外的所有被试预训练 ────────────────────
            pretrain_ids = [s for s in subject_ids if s != sub_idx]
            trainer      = ExGAN(sub_idx, DATA_DIR, LOG_DIR, seq_len)
            trainer.pretrain(
                pretrain_subject_ids = pretrain_ids,
                save_path            = PRETRAIN_PATH,
                n_epochs             = 30,
                batch_size           = 128
            )

            # ── 阶段二：加载预训练权重，对本被试做5折微调 ────────────────
            print(f"\n  [微调阶段] Subject HC{sub_idx}")
            best5, aver5 = 0.0, 0.0
            for fold in range(5):
                print(f"\n  {'─'*45}")
                print(f"  Fold {fold+1}/5")
                print(f"  {'─'*45}")
                # 每折都从预训练权重重新加载，保证各折独立
                exgan = ExGAN(sub_idx, DATA_DIR, LOG_DIR, seq_len)
                exgan.load_pretrain(PRETRAIN_PATH)
                ba, aa, _, _ = exgan.finetune(
                    fold, n_epochs=60, batch_size=64
                )
                result_write.write(
                    f"  Fold {fold+1}: best={ba:.4f}  aver={aa:.4f}\n"
                )
                best5 += ba
                aver5 += aa

            best5 /= 5
            aver5 /= 5
            result_write.write(
                f"  5-fold best={best5:.4f}  aver={aver5:.4f}\n"
            )
            result_write.write("-" * 55 + "\n")
            result_write.flush()

            best_all += best5
            aver_all += aver5
            elapsed = datetime.datetime.now() - starttime
            print(f"\nHC{sub_idx} 总耗时: {elapsed}")

        best_all /= n_subjects
        aver_all /= n_subjects
        result_write.write(
            f"\nAll subjects  best={best_all:.4f}  aver={aver_all:.4f}\n"
        )
        print(f"\n{'='*55}")
        print(f"All subjects  best={best_all:.4f}  aver={aver_all:.4f}")


if __name__ == "__main__":
    print(time.asctime())
    main()
    print(time.asctime())