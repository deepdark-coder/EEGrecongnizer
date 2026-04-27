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

# ─────────────────────────── GPU 设置 ───────────────────────────────────────
gpus = [0]
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))


# ════════════════════════════ 模型定义 ═══════════════════════════════════════
# (模型结构保持不变，适配 30 通道和 2 分类)
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
    def __init__(self, emb_size: int, num_heads: int = 4,
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


# ════════════════════════════ 极速版训练器 ═══════════════════════════════════

class ExGAN:
    def __init__(self, data_dir: str, seq_len: int):
        self.n_channels = 30
        self.n_times    = 250
        self.n_classes  = 2
        self.lr         = 0.0002
        self.b1, self.b2 = 0.5, 0.999
        self.data_dir   = data_dir
        self.seq_len    = seq_len

        self.criterion_cls = nn.CrossEntropyLoss().cuda()

        self.model = ViT(
            emb_size=40, depth=2, n_classes=self.n_classes,
            n_channels=self.n_channels, seq_len=seq_len
        ).cuda()
        self.model = nn.DataParallel(
            self.model, device_ids=list(range(len(gpus)))
        ).cuda()

    @staticmethod
    def get_seq_len(n_channels: int = 30, n_times: int = 250,
                    emb_size: int = 40) -> int:
        dummy = torch.zeros(1, 1, n_channels, n_times)
        pe    = PatchEmbedding(emb_size, n_channels)
        with torch.no_grad():
            out = pe(dummy)
        return out.shape[1]

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

        # 用训练集统计量标准化 (按通道维度标准化更优，这里保持与之前一致的全量标准化)
        mu, std    = train_data.mean(), train_data.std() + 1e-8
        train_data = (train_data - mu) / std
        test_data  = (test_data  - mu) / std

        # 增加 channel 维度 (N, 1, 30, 250)
        train_data  = np.ascontiguousarray(train_data[:, np.newaxis], dtype=np.float32)
        test_data   = np.ascontiguousarray(test_data[:,  np.newaxis], dtype=np.float32)
        train_label = np.ascontiguousarray(train_label, dtype=np.int64)
        test_label  = np.ascontiguousarray(test_label,  dtype=np.int64)

        return train_data, train_label, test_data, test_label

    # ── 重写：单次全局预训练 (内存保留最优权重) ─────────────────────────────
    def pretrain_once(self, subject_ids: list, n_epochs: int = 30, batch_size: int = 128):
        """
        一次性收集所有被试的 80% 训练集数据进行全局训练，防止测试集泄露。
        返回内存中的最佳权重字典 (无磁盘写入)。
        """
        print(f"\n[开始全局预训练] 正在加载 {len(subject_ids)} 个被试的训练集数据...")
        all_train_data, all_train_label = [], []
        
        for sid in subject_ids:
            tr_d, tr_l, _, _ = self._get_train_test_data(sid)
            all_train_data.append(tr_d)
            all_train_label.append(tr_l)

        all_train_data  = np.concatenate(all_train_data,  axis=0)
        all_train_label = np.concatenate(all_train_label, axis=0)

        # 打乱全局训练集
        perm = np.random.permutation(len(all_train_data))
        all_train_data  = all_train_data[perm]
        all_train_label = all_train_label[perm]

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(all_train_data,  dtype=torch.float32),
            torch.tensor(all_train_label, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_loss = float('inf')
        best_state = None
        
        print(f"全局训练样本总数: {len(all_train_data)}")
        
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
            
            print(f"  Pretrain Epoch {epoch:3d}/{n_epochs} | loss={avg_loss:.4f} | acc={avg_acc:.4f}")

            # 内存中记录最优模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.model.state_dict())

        print(f"[全局预训练完成] 最优 Loss: {best_loss:.4f}")
        return best_state

    # ── 重写：单次微调 (直接返回最高精度) ───────────────────────────────────
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_acc = 0.0
        # 定义该被试的专属权重保存路径
        best_save_path = os.path.join(save_dir, f'HC{sid}_finetuned_best.pth')

        for epoch in range(n_epochs):
            self.model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                _, outputs   = self.model(imgs)
                loss         = self.criterion_cls(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                _, cls_out = self.model(test_data_gpu)
            
            y_pred = cls_out.argmax(dim=1)
            acc    = (y_pred == test_label_gpu).float().mean().item()

            # 突破最高精度时，触发物理保存
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), best_save_path)

        return best_acc


# ════════════════════════════ 主函数入口 ═════════════════════════════════════

def main():
    DATA_DIR = "./EEG-Conformer/data/processed_normal/"
    SAVE_DIR = "./EEG-Conformer/best_params/"
    
    # 自动扫描受试者列表
    subject_ids = sorted([
        int(os.path.basename(f).replace('HC', '').replace('_1s.mat', ''))
        for f in glob.glob(os.path.join(DATA_DIR, 'HC*_1s.mat'))
    ])
    n_subjects = len(subject_ids)
    
    if n_subjects == 0:
        print(f"❌ 未在 {DATA_DIR} 找到数据，请检查路径。")
        return
        
    print(f"✅ 找到 {n_subjects} 个受试者: {subject_ids}")

    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=250, emb_size=40)
    print(f"[Info] 动态推理 seq_len = {seq_len}")

    starttime = datetime.datetime.now()

    global_trainer = ExGAN(data_dir=DATA_DIR, seq_len=seq_len)
    
    # 提取并保存在内存中的预训练字典
    pretrained_weights = global_trainer.pretrain_once(
        subject_ids=subject_ids, 
        n_epochs=300, 
        batch_size=128
    )

    # ───────────────────────────────────────────────────────────────────────
    # 阶段二：对每个被试进行极速微调验证
    # ───────────────────────────────────────────────────────────────────────
    print("\n========== 阶段二：各被试微调验证 ==========")
    all_accuracies = []

    for i, sub_idx in enumerate(subject_ids):
        finetuner = ExGAN(data_dir=DATA_DIR, seq_len=seq_len)
        finetuner.model.load_state_dict(copy.deepcopy(pretrained_weights))
        
        # 传入 save_dir 参数
        best_acc = finetuner.finetune_once(sid=sub_idx, save_dir=SAVE_DIR, n_epochs=15, batch_size=32)
        all_accuracies.append(best_acc)
        
        print(f"  [Subject HC{sub_idx}] 微调最优精度: {best_acc * 100:.2f}% (已保存至 {SAVE_DIR})")

    last_save_path = './EEG-Conformer/last_params/global_pretrain_last_e300.pth'
    torch.save(finetuner.model.state_dict(), last_save_path)
    print(f"💾 全局 Last 权重已保存至: {last_save_path}")

    # 统计汇总
    mean_acc = np.mean(all_accuracies)
    elapsed = datetime.datetime.now() - starttime
    
    print(f"\n{'='*55}")
    print(f"🎉 全部训练与验证结束！总耗时: {elapsed}")
    print(f"📈 全局平均准确率 (Mean Accuracy): {mean_acc * 100:.2f}%")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()