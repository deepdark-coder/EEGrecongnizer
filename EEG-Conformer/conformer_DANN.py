import os, random, datetime, time, glob, copy
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.backends import cudnn
from torch.autograd import Function
from einops import rearrange
from einops.layers.torch import Rearrange

cudnn.benchmark     = False
cudnn.deterministic = True

# 硬件设备设置
gpus = [0]
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))


# ============================ 梯度反转层 (GRL) ============================
class GradientReversalLayer(Function):
    """
    梯度反转层：在前向传播时保持输入不变，在反向传播时将梯度乘以一个负数(alpha)。
    这是实现领域对抗网络(DANN)的核心组件。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# ============================ 模型基础组件 ============================
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 24, n_channels: int = 30):
        super().__init__()
        self.shallownet = nn.Sequential(
            # 1. 频率点独立映射：使用 (1, 1) 卷积核，保证每个频率点(1-50Hz)独立计算，不混合
            nn.Conv2d(1, 40, (1, 1), stride=(1, 1)),
            
            # 2. 空间特征提取：在 30 个物理通道之间进行空间特征融合
            nn.Conv2d(40, 40, (n_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            
            # 注意：此处已移除所有的池化层(AvgPool2d)，保留完整的 50 维高分辨率频率序列
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


# ============================ 预测头与主网络 ============================
class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()
        # 接收 GAP 后的特征，输入维度仅为 emb_size
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 32),
            nn.ELU(),
            nn.Dropout(0.5), # 提高 Dropout 防止记忆效应
            nn.Linear(32, n_classes),
        )
    def forward(self, x: Tensor):
        return x, self.fc(x)

class ViT(nn.Module):
    def __init__(self, emb_size: int = 24, depth: int = 2,
                 n_classes: int = 2, n_channels: int = 30, n_subjects: int = 60):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels)
        self.transformer     = TransformerEncoder(depth, emb_size)
        self.cls_head        = ClassificationHead(emb_size, n_classes)
        
        # 领域判别头 (用于识别样本属于哪一个被试)
        self.domain_head = nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_subjects)
        )

    def forward(self, x: Tensor, alpha: float = 1.0):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        
        # 全局平均池化 (GAP)：将频率维度的特征平均化，提取位置无关的全局特征，防止过拟合
        feat = x.mean(dim=1)
        
        # 主任务分支：情绪分类
        _, cls_out = self.cls_head(feat)
        
        # 对抗任务分支：受试者身份判别 (梯度反转)
        reverse_feat = GradientReversalLayer.apply(feat, alpha)
        domain_out = self.domain_head(reverse_feat)
        
        return cls_out, domain_out


# ============================ 训练与验证引擎 ============================
class ExGAN:
    def __init__(self, data_dir: str, seq_len: int, depth: int, emb_size: int, n_subjects: int):
        self.n_channels = 30
        self.n_times    = 250
        self.n_classes  = 2
        self.n_subjects = n_subjects
        self.lr         = 0.0002
        self.b1, self.b2 = 0.5, 0.999
        self.data_dir   = data_dir
        self.seq_len    = seq_len
        self.depth      = depth
        self.emb_size   = emb_size

        self.criterion_cls    = nn.CrossEntropyLoss().cuda()
        self.criterion_domain = nn.CrossEntropyLoss().cuda()

        self.model = ViT(
            emb_size=self.emb_size, depth=self.depth, n_classes=self.n_classes,
            n_channels=self.n_channels, n_subjects=self.n_subjects
        ).cuda()
        self.model = nn.DataParallel(
            self.model, device_ids=list(range(len(gpus)))
        ).cuda()

    @staticmethod
    def extract_log_psd(data, fs=250):
        """
        物理特征提取：提取 1-50Hz 的高分辨率功率谱密度 (Log-PSD)
        """
        trials, channels, timepoints = data.shape
        fft_data = np.fft.fft(data, axis=2)
        power_spectrum = (np.abs(fft_data) ** 2) / timepoints
        freqs = np.fft.fftfreq(timepoints, 1/fs)
        
        idx = np.where((freqs >= 1) & (freqs <= 50))[0]
        psd_features = power_spectrum[:, :, idx]
        return np.log10(psd_features + 1e-8)

    @staticmethod
    def get_seq_len(n_channels: int = 30, n_times: int = 50, emb_size: int = 24) -> int:
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

        # 核心预处理：将时域信号转换为 50 维频率特征
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

        # 数据标准化
        mu, std    = train_data.mean(), train_data.std() + 1e-8
        train_data = (train_data - mu) / std
        test_data  = (test_data  - mu) / std

        # 增加通道维度以匹配网络输入 (N, 1, 30, 50)
        train_data  = np.ascontiguousarray(train_data[:, np.newaxis], dtype=np.float32)
        test_data   = np.ascontiguousarray(test_data[:,  np.newaxis], dtype=np.float32)
        train_label = np.ascontiguousarray(train_label, dtype=np.int64)
        test_label  = np.ascontiguousarray(test_label,  dtype=np.int64)

        return train_data, train_label, test_data, test_label

    def pretrain_once(self, subject_ids: list, save_dir: str, n_epochs: int = 40, batch_size: int = 128, patience: int = 10):
        print(f"\n[DANN 全局预训练] 正在加载 {len(subject_ids)} 个受试者的数据并构建对抗网络数据集...")
        all_train_data, all_train_label, all_train_domain = [], [], []
        all_val_data, all_val_label, all_val_domain = [], [], []
        
        # 将实际的受试者编号映射为 0 到 N-1 的标签，以适配领域分类器的交叉熵
        for domain_label, sid in enumerate(subject_ids):
            tr_d, tr_l, val_d, val_l = self._get_train_test_data(sid)
            
            all_train_data.append(tr_d)
            all_train_label.append(tr_l)
            all_train_domain.append(np.full(tr_l.shape, domain_label, dtype=np.int64))
            
            all_val_data.append(val_d)
            all_val_label.append(val_l)
            all_val_domain.append(np.full(val_l.shape, domain_label, dtype=np.int64))

        all_train_data   = np.concatenate(all_train_data,   axis=0)
        all_train_label  = np.concatenate(all_train_label,  axis=0)
        all_train_domain = np.concatenate(all_train_domain, axis=0)
        
        all_val_data   = np.concatenate(all_val_data,   axis=0)
        all_val_label  = np.concatenate(all_val_label,  axis=0)
        all_val_domain = np.concatenate(all_val_domain, axis=0)

        # 仅打乱训练集
        perm = np.random.permutation(len(all_train_data))
        all_train_data   = all_train_data[perm]
        all_train_label  = all_train_label[perm]
        all_train_domain = all_train_domain[perm]

        # 构建包含 图像、情绪标签、领域标签 的三元组数据集
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(all_train_data,   dtype=torch.float32),
            torch.tensor(all_train_label,  dtype=torch.long),
            torch.tensor(all_train_domain, dtype=torch.long)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(all_val_data,   dtype=torch.float32),
            torch.tensor(all_val_label,  dtype=torch.long),
            torch.tensor(all_val_domain, dtype=torch.long)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_acc = 0.0
        patience_counter = 0
        best_save_path = os.path.join(save_dir, f'D{self.depth}_H4_S{self.emb_size}_DANN_best.pth')
        
        print(f"全局训练集样本数: {len(all_train_data)} | 全局验证集样本数: {len(all_val_data)}")
        
        for epoch in range(n_epochs):
            # 动态计算 DANN 的对抗强度 alpha (训练初期较小，后期增大至接近 1.0)
            p = float(epoch) / n_epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1.

            self.model.train()
            train_cls_loss, train_dom_loss, train_correct = 0.0, 0.0, 0
            
            for imgs, labels, domains in train_loader:
                imgs, labels, domains = imgs.cuda(), labels.cuda(), domains.cuda()
                
                # 前向传播，输出情绪预测与领域预测
                cls_out, domain_out = self.model(imgs, alpha=alpha)
                
                loss_cls    = self.criterion_cls(cls_out, labels)
                loss_domain = self.criterion_domain(domain_out, domains)
                
                # 核心修复：对 domain_loss 进行降维打击。
                # 权重设为 0.05 到 0.1 之间，确保分类任务绝对主导，对抗任务只做微调
                domain_weight = 0.05 
                
                # 总损失 = 情绪分类损失 + 极小权重的对抗损失
                loss = loss_cls + domain_weight * loss_domain
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_cls_loss += loss_cls.item() * len(imgs)
                train_dom_loss += loss_domain.item() * len(imgs)
                train_correct  += (cls_out.argmax(1) == labels).sum().item()

            scheduler.step()
            avg_train_loss = train_cls_loss / len(all_train_data)
            avg_train_acc  = train_correct / len(all_train_data)

            self.model.eval()
            val_loss, val_correct = 0.0, 0
            
            with torch.no_grad():
                for v_imgs, v_labels, v_domains in val_loader:
                    v_imgs, v_labels, v_domains = v_imgs.cuda(), v_labels.cuda(), v_domains.cuda()
                    # 验证阶段关闭对抗，alpha 设为 0 (仅作为安全阻断)
                    v_cls_out, _ = self.model(v_imgs, alpha=0.0)
                    v_loss_cls   = self.criterion_cls(v_cls_out, v_labels)
                    
                    val_loss    += v_loss_cls.item() * len(v_imgs)
                    val_correct += (v_cls_out.argmax(1) == v_labels).sum().item()

            avg_val_loss = val_loss / len(all_val_data)
            avg_val_acc  = val_correct / len(all_val_data)
            
            print(f"  Epoch {epoch+1:2d}/{n_epochs} [Alpha: {alpha:.2f}] | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}")

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                torch.save(self.model.state_dict(), best_save_path)
                patience_counter = 0
                print(f"    最佳验证集精度更新。权重已保存至: {best_save_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"连续 {patience} 个 Epoch 验证集精度未提升，触发早停机制，终止预训练。")
                break

        print(f"\n[全局预训练结束] 历史最高验证集精度 (Best Val Acc): {best_val_acc * 100:.2f}%")
        return best_save_path

    def finetune_once(self, sid: int, save_dir: str, n_epochs: int = 15, batch_size: int = 32):
        """
        单受试者微调逻辑。在微调阶段不关注领域分类，仅针对分类分支进行微调。
        """
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
                # 微调时不需要对抗损失，alpha 设置为 0
                cls_out, _   = self.model(imgs, alpha=0.0)
                loss         = self.criterion_cls(cls_out, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                cls_out, _ = self.model(test_data_gpu, alpha=0.0)
            
            y_pred = cls_out.argmax(dim=1)
            acc    = (y_pred == test_label_gpu).float().mean().item()

            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), best_save_path)

        return best_acc


# ============================ 启动入口 ============================
def main():
    DATA_DIR = "./EEG-Conformer/data/processed_normal/"
    SAVE_DIR = "./EEG-Conformer/last_params/"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    emb_size = 24
    depth    = 2
    
    # 自动扫描并提取受试者列表
    subject_ids = sorted([
        int(os.path.basename(f).replace('HC', '').replace('_1s.mat', ''))
        for f in glob.glob(os.path.join(DATA_DIR, 'HC*_1s.mat'))
    ])
    n_subjects = len(subject_ids)
    
    if n_subjects == 0:
        print(f"未在 {DATA_DIR} 找到数据，请检查路径配置。")
        return
        
    print(f"成功扫描到 {n_subjects} 个受试者数据。")

    # 注意：此处传入 50，代表 50 维的高分辨率频点特征
    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=50, emb_size=emb_size)
    print(f"动态推断出的序列长度 seq_len = {seq_len}")

    starttime = datetime.datetime.now()

    # 初始化训练器，传入受试者总数用于构建判别器
    global_trainer = ExGAN(
        data_dir=DATA_DIR, 
        seq_len=seq_len, 
        depth=depth, 
        emb_size=emb_size,
        n_subjects=n_subjects
    )
    
    # 开始执行带全局验证机制的 DANN 预训练
    best_weights_path = global_trainer.pretrain_once(
        subject_ids=subject_ids, 
        save_dir=SAVE_DIR,
        n_epochs=150,       
        batch_size=128,
        patience=100        
    )
    
    elapsed = datetime.datetime.now() - starttime
    print(f"全局预训练执行完毕，总耗时: {elapsed}")


if __name__ == "__main__":
    main()