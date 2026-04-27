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

gpus = [0]
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 24, n_channels: int = 30):
        super().__init__()
        self.shallownet = nn.Sequential(
            # 【关键修改 1】：(1, 1) 的卷积核。
            # 意味着每一个频率点（1Hz, 2Hz...）只和自己做线性映射，绝不和相邻的频率混合！
            nn.Conv2d(1, 40, (1, 1), stride=(1, 1)),
            
            # 空间卷积：在 30 个电极之间寻找空间联系（这步保留，很重要）
            nn.Conv2d(40, 40, (n_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            
            # 【关键修改 2】：删掉所有的 AvgPool2d！
            # 坚决不降维，我们要把 50 个高分辨率频率点完整送进 Transformer！
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
            # 【修改点 1】：去掉 seq_len * 的乘法，直接接收 emb_size 维度的输入
            nn.Linear(emb_size, 32),
            nn.ELU(),
            nn.Dropout(0.5), # 顺手把这里的 Dropout 从 0.3 提高到 0.5，进一步抑制过拟合
            nn.Linear(32, n_classes),
        )
    def forward(self, x: Tensor):
        # x 的输入形状: (Batch, seq_len, emb_size)
        
        # 【修改点 2】：用 Global Average Pooling (GAP) 替代原来的 view 展平操作
        # 对 seq_len 维度求平均，输出形状变为 (Batch, emb_size)
        feat = x.mean(dim=1) 
        
        return feat, self.fc(feat)

class ViT(nn.Module):
    def __init__(self, emb_size: int = 16, depth: int = 2,
                 n_classes: int = 2, n_channels: int = 30, seq_len: int = 11):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels)
        self.transformer     = TransformerEncoder(depth, emb_size)
        self.cls_head        = ClassificationHead(emb_size, n_classes, seq_len)

    def forward(self, x: Tensor):
        x = self.patch_embedding(x)
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
        self.emb_size   = emb_size

        self.criterion_cls = nn.CrossEntropyLoss().cuda()

        self.model = ViT(
            emb_size=self.emb_size, depth=self.depth, n_classes=self.n_classes,
            n_channels=self.n_channels, seq_len=seq_len
        ).cuda()
        self.model = nn.DataParallel(
            self.model, device_ids=list(range(len(gpus)))
        ).cuda()

    # MODIFICATION 3: Added static method for DE feature extraction.
    @staticmethod
    def extract_log_psd(data, fs=250):
        trials, channels, timepoints = data.shape
        fft_data = np.fft.fft(data, axis=2)
        power_spectrum = (np.abs(fft_data) ** 2) / timepoints
        freqs = np.fft.fftfreq(timepoints, 1/fs)
        
        idx = np.where((freqs >= 1) & (freqs <= 50))[0]
        psd_features = power_spectrum[:, :, idx]
        log_psd = np.log10(psd_features + 1e-8)
        
        # 核心改动：Trial-wise Instance Normalization
        # 沿着频率维度 (axis=2) 求均值和标准差
        # 彻底抹平个体头骨厚度、设备阻抗带来的绝对能量差异，只保留相对形态！
        mu = np.mean(log_psd, axis=2, keepdims=True)
        std = np.std(log_psd, axis=2, keepdims=True) + 1e-8
        log_psd = (log_psd - mu) / std
        
        return log_psd

    @staticmethod
    def get_seq_len(n_channels: int = 30, n_times: int = 250,
                    emb_size: int = 16) -> int:
        dummy = torch.zeros(1, 1, n_channels, n_times)
        pe    = PatchEmbedding(emb_size, n_channels)
        with torch.no_grad():
            out = pe(dummy)
        return out.shape[1]

    def _get_train_test_data(self, sid: int):
        mat_file = os.path.join(self.data_dir, f'HC{sid}_1s.mat')
        mat      = scipy.io.loadmat(mat_file)

        all_data  = np.ascontiguousarray(mat['data'],            dtype=np.float32)
        all_label = np.ascontiguousarray(mat['label'].flatten(), dtype=np.int64)

        # MODIFICATION 4: Call extract_de_features before data split and Z-score.
        # This converts all_data from (400, 30, 250) to (400, 30, 5)
        all_data = self.extract_log_psd(all_data, fs=250)

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

        train_data  = np.ascontiguousarray(train_data[:, np.newaxis], dtype=np.float32)
        test_data   = np.ascontiguousarray(test_data[:,  np.newaxis], dtype=np.float32)
        train_label = np.ascontiguousarray(train_label, dtype=np.int64)
        test_label  = np.ascontiguousarray(test_label,  dtype=np.int64)

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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_acc = 0.0
        patience_counter = 0
        best_save_path = os.path.join(save_dir, f'D{self.depth}_H4_S{self.emb_size}_best1.pth')
        
        print(f" 全局训练集大小: {len(all_train_data)} | 全局验证集大小: {len(all_val_data)}")
        
        for epoch in range(n_epochs):
            self.model.train()
            train_loss, train_correct = 0.0, 0
            
            for imgs, labels in train_loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                _, outputs   = self.model(imgs)
                loss         = self.criterion_cls(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss    += loss.item() * len(imgs)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            scheduler.step()
            avg_train_loss = train_loss / len(all_train_data)
            avg_train_acc  = train_correct / len(all_train_data)

            self.model.eval()
            val_loss, val_correct = 0.0, 0
            
            with torch.no_grad():
                for v_imgs, v_labels in val_loader:
                    v_imgs, v_labels = v_imgs.cuda(), v_labels.cuda()
                    _, v_outputs     = self.model(v_imgs)
                    v_loss           = self.criterion_cls(v_outputs, v_labels)
                    
                    val_loss    += v_loss.item() * len(v_imgs)
                    val_correct += (v_outputs.argmax(1) == v_labels).sum().item()

            avg_val_loss = val_loss / len(all_val_data)
            avg_val_acc  = val_correct / len(all_val_data)
            
            print(f"  Epoch {epoch+1:2d}/{n_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}")

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                torch.save(self.model.state_dict(), best_save_path)
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_acc = 0.0
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

            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), best_save_path)

        return best_acc

def main():
    DATA_DIR = "./EEG-Conformer/data/processed_normal/"
    SAVE_DIR = "./EEG-Conformer/last_params/"
    emb_size = 24
    depth   = 2
    
    subject_ids = sorted([
        int(os.path.basename(f).replace('HC', '').replace('_1s.mat', ''))
        for f in glob.glob(os.path.join(DATA_DIR, 'HC*_1s.mat'))
    ])
    n_subjects = len(subject_ids)
    
    if n_subjects == 0:
        print(f"未在 {DATA_DIR} 找到数据，请检查路径。")
        return
        
    print(f"找到 {n_subjects} 个受试者: {subject_ids}")

    # MODIFICATION 5: Changed n_times to 5 during dynamic get_seq_len calculation.
    # The input to the Transformer is now based on 5 frequency bands instead of 250 timepoints.
    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=50, emb_size=16)
    print(f"[Info] 动态推理 seq_len = {seq_len}")

    starttime = datetime.datetime.now()

    global_trainer = ExGAN(data_dir=DATA_DIR, seq_len=seq_len, depth=depth, emb_size=emb_size)
    
    best_weights_path = global_trainer.pretrain_once(
        subject_ids=subject_ids, 
        save_dir=SAVE_DIR,
        n_epochs=150,       
        batch_size=128,
        patience=30        
    )

if __name__ == "__main__":
    main()