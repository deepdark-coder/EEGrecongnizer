import os
import glob
import datetime
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.backends import cudnn
from einops import rearrange
from einops.layers.torch import Rearrange
from sklearn.model_selection import KFold


cudnn.benchmark = False
cudnn.deterministic = True

gpus = [1]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))


def denoise_signals(all_data):
    std_val = np.std(all_data)
    threshold = 3 * std_val
    all_data = np.clip(all_data, -threshold, threshold)
    common_mode_noise = np.mean(all_data, axis=1, keepdims=True)
    all_data = all_data - common_mode_noise
    return all_data


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
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.shallownet(x))


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


class ConvolutionModule(nn.Module):
    def __init__(self, emb_size: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(emb_size)
        self.pointwise_conv1 = nn.Conv1d(emb_size, emb_size * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            emb_size,
            emb_size,
            kernel_size,
            padding=kernel_size // 2,
            groups=emb_size,
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
    def __init__(
        self,
        emb_size: int,
        num_heads: int = 4,
        drop_p: float = 0.1,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.1,
        drop_path: float = 0.0,
        conv_kernel: int = 31,
    ):
        super().__init__()
        self.ffn1 = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
                DropPath(drop_path),
            )
        )
        self.mhsa = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
                DropPath(drop_path),
            )
        )
        self.conv = ConvolutionModule(emb_size, conv_kernel, drop_p)
        self.ffn2 = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
                DropPath(drop_path),
            )
        )
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
        super().__init__(*[ConformerBlock(emb_size, drop_path=drop_path_rates[i]) for i in range(depth)])


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
    def __init__(self, emb_size: int, depth: int, n_classes: int = 2, n_channels: int = 30, seq_len: int = 11):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, emb_size) * 0.02)
        self.transformer = ConformerEncoder(depth, emb_size)
        self.cls_head = ClassificationHead(emb_size, n_classes)

    def forward(self, x: Tensor):
        x = self.patch_embedding(x)
        x = x + self.pos_embedding
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
    def __init__(self, data_dir: str, seq_len: int, depth: int, emb_size: int):
        self.n_channels = 30
        self.n_times = 250
        self.n_classes = 2
        self.lr = 0.0002
        self.b1, self.b2 = 0.5, 0.999
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.depth = depth
        self.emb_size = emb_size

        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
        self.model = ViT(
            emb_size=self.emb_size,
            depth=self.depth,
            n_classes=self.n_classes,
            n_channels=self.n_channels,
            seq_len=seq_len,
        ).cuda()
        self.model = nn.DataParallel(self.model, device_ids=list(range(len(gpus)))).cuda()

    @staticmethod
    def get_seq_len(n_channels: int = 30, n_times: int = 250, emb_size: int = 16) -> int:
        dummy = torch.zeros(1, 1, n_channels, n_times)
        pe = PatchEmbedding(emb_size, n_channels)
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

    def _load_subject_data(self, sid: int):
        mat_file = os.path.join(self.data_dir, f"HC{sid}_1s.mat")
        mat = scipy.io.loadmat(mat_file)

        all_data = np.ascontiguousarray(mat["data"], dtype=np.float32)
        all_label = np.ascontiguousarray(mat["label"].flatten(), dtype=np.int64)
        all_data = denoise_signals(all_data)

        return all_data, all_label

    def _normalize_with_train_stats(self, train_data, eval_data):
        mu = train_data.mean(axis=(0, 2), keepdims=True)
        std = train_data.std(axis=(0, 2), keepdims=True) + 1e-8
        train_data = (train_data - mu) / std
        eval_data = (eval_data - mu) / std
        return train_data, eval_data

    def _prepare_fold_data(self, train_val_subject_ids, test_subject_ids, val_ratio=0.125, fold_seed=42):
        train_val_subject_ids = np.array(train_val_subject_ids)
        n_train_val = len(train_val_subject_ids)
        n_val = max(1, int(n_train_val * val_ratio))
        val_pick = np.random.RandomState(fold_seed).choice(np.arange(n_train_val), n_val, replace=False)
        val_subject_ids = train_val_subject_ids[val_pick].tolist()
        train_subject_ids = np.delete(train_val_subject_ids, val_pick).tolist()

        train_data_list = []
        train_label_list = []
        for sid in train_subject_ids:
            data, label = self._load_subject_data(int(sid))
            train_data_list.append(data)
            train_label_list.append(label)

        train_base = np.concatenate(train_data_list, axis=0)
        train_concat = train_base.copy()
        label_concat = np.concatenate(train_label_list, axis=0)
        train_concat, _ = self._normalize_with_train_stats(train_base, train_concat)
        train_concat = np.ascontiguousarray(train_concat[:, np.newaxis], dtype=np.float32)
        label_concat = np.ascontiguousarray(label_concat, dtype=np.int64)

        val_data_list = []
        val_label_list = []
        for sid in val_subject_ids:
            data, label = self._load_subject_data(int(sid))
            _, data = self._normalize_with_train_stats(train_base, data)
            val_data_list.append(data)
            val_label_list.append(label)

        test_data_list = []
        test_label_list = []
        for sid in test_subject_ids:
            data, label = self._load_subject_data(int(sid))
            _, data = self._normalize_with_train_stats(train_base, data)
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

        return (
            train_concat,
            label_concat,
            val_concat,
            val_label_concat,
            test_concat,
            test_label_concat,
            train_subject_ids,
            val_subject_ids,
        )

    def run_subject_cv(self, subject_ids, save_dir, n_epochs=300, batch_size=128, patience=60, seed=42, val_ratio=0.125, n_tta=5):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_results = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(np.array(subject_ids)), 1):
            train_val_subjects = np.array(subject_ids)[train_val_idx].tolist()
            test_subjects = np.array(subject_ids)[test_idx].tolist()

            (
                train_data,
                train_label,
                val_data,
                val_label,
                test_data,
                test_label,
                train_subjects,
                val_subjects,
            ) = self._prepare_fold_data(
                train_val_subjects,
                test_subjects,
                val_ratio=val_ratio,
                fold_seed=42 + fold_idx,
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
            )
            test_data_gpu = torch.tensor(test_data, dtype=torch.float32).cuda()
            test_label_gpu = torch.tensor(test_label, dtype=torch.long).cuda()

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=5e-4)
            scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=n_epochs)
            ema = EMA(self.model, decay=0.999)

            best_val_acc = 0.0
            patience_counter = 0
            best_save_path = os.path.join(save_dir, f"subjectcv_fold{fold_idx}_best.pth")

            for epoch in range(n_epochs):
                self.model.train()
                train_loss = 0.0
                train_correct = 0

                for imgs, labels in train_loader:
                    imgs, labels = imgs.cuda(), labels.cuda()
                    imgs = ExGAN.augment(imgs)
                    if torch.rand(1).item() < 0.5:
                        imgs, labels_a, labels_b, lam = ExGAN.mixup(imgs, labels, alpha=0.1)
                        _, outputs = self.model(imgs)
                        loss = lam * self.criterion_cls(outputs, labels_a) + (1 - lam) * self.criterion_cls(outputs, labels_b)
                    else:
                        _, outputs = self.model(imgs)
                        loss = self.criterion_cls(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    ema.update()

                    train_loss += loss.item() * len(imgs)
                    train_correct += (outputs.argmax(1) == labels).sum().item()

                scheduler.step()
                avg_train_loss = train_loss / len(train_label)
                avg_train_acc = train_correct / len(train_label)

                self.model.eval()
                ema.apply_shadow()
                val_loss = 0.0
                val_correct = 0
                with torch.no_grad():
                    for v_imgs, v_labels in val_loader:
                        v_imgs, v_labels = v_imgs.cuda(), v_labels.cuda()
                        _, v_outputs = self.model(v_imgs)
                        v_loss = self.criterion_cls(v_outputs, v_labels)
                        val_loss += v_loss.item() * len(v_imgs)
                        val_correct += (v_outputs.argmax(1) == v_labels).sum().item()
                ema.restore()

                avg_val_loss = val_loss / len(val_label)
                avg_val_acc = val_correct / len(val_label)

                print(
                    f"Epoch {epoch + 1:2d}/{n_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}"
                )

                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    ema.apply_shadow()
                    torch.save(self.model.state_dict(), best_save_path)
                    ema.restore()
                    patience_counter = 0
                    print(f"new best: save to: {best_save_path}")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping after {patience} epochs without val improvement.")
                    break

            self.model.load_state_dict(torch.load(best_save_path, map_location="cuda"), strict=True)

            self.model.eval()
            with torch.no_grad():
                cls_out = self.tta_evaluate(test_data_gpu, n_views=n_tta)

            y_pred = cls_out.argmax(dim=1)
            test_acc = (y_pred == test_label_gpu).float().mean().item()
            fold_results.append(test_acc)
            print(f"Fold {fold_idx} TTA-{n_tta} Test Acc: {test_acc * 100:.2f}% (best val: {best_val_acc * 100:.2f}%)")

        print(f"\n{'=' * 60}")
        print(f"CV Results: {[f'{acc * 100:.2f}' for acc in fold_results]}")
        print(f"Mean: {np.mean(fold_results) * 100:.2f}% +- {np.std(fold_results) * 100:.2f}%")
        print(f"{'=' * 60}")


def read_data(data_dir: str):
    subject_ids = sorted(
        [
            int(os.path.basename(f).replace("HC", "").replace("_1s.mat", ""))
            for f in glob.glob(os.path.join(data_dir, "HC*_1s.mat"))
        ]
    )
    return subject_ids


def main():
    data_dir = "./EEG-Conformer/data/processed_normal/"
    save_dir = "./EEG-Conformer/cluster_params/"
    emb_size = 40
    depth = 2

    subject_ids = read_data(data_dir)
    n_subjects = len(subject_ids)
    if n_subjects == 0:
        print(f"No data found under {data_dir}")
        return

    print(f"Found {n_subjects} subjects: {subject_ids}")
    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=250, emb_size=emb_size)
    print(f"[Info] seq_len = {seq_len}")

    starttime = datetime.datetime.now()
    trainer = ExGAN(data_dir=data_dir, seq_len=seq_len, depth=depth, emb_size=emb_size)
    trainer.run_subject_cv(
        subject_ids=subject_ids,
        save_dir=save_dir,
        n_epochs=300,
        batch_size=128,
        patience=60,
        seed=42,
        val_ratio=0.125,
        n_tta=5,
    )
    print(f"Total elapsed: {datetime.datetime.now() - starttime}")


if __name__ == "__main__":
    main()
