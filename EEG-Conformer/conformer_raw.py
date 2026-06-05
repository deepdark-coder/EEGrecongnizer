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


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 24, n_channels: int = 30):
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
        att = self.att_drop(F.softmax(energy / self.emb_size ** 0.5, dim=-1))
        out = torch.einsum("bhal, bhlv -> bhav", att, v)
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
    def __init__(
        self,
        emb_size: int,
        num_heads: int = 4,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
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
    def __init__(
        self,
        emb_size: int = 24,
        depth: int = 2,
        n_classes: int = 2,
        n_channels: int = 30,
        seq_len: int = 11,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels)
        self.transformer = TransformerEncoder(depth, emb_size)
        self.cls_head = ClassificationHead(emb_size, n_classes, seq_len)

    def forward(self, x: Tensor):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        return self.cls_head(x)


def warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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


class ExGAN:
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        depth: int,
        emb_size: int,
        device: torch.device,
        lr: float = 2e-4,
        weight_decay: float = 5e-4,
        model_seed: int = 42,
        use_denoise: bool = False,
        amp_enabled: bool = True,
        aug_noise_std: float = 0.01,
        aug_shift: int = 8,
        channel_mask_prob: float = 0.0,
        channel_drop_prob: float = 0.0,
        mixup_prob: float = 0.0,
        mixup_alpha: float = 0.1,
        label_smoothing: float = 0.05,
    ):
        self.n_channels = 30
        self.n_times = 250
        self.n_classes = 2
        self.lr = lr
        self.b1, self.b2 = 0.5, 0.999
        self.weight_decay = weight_decay
        self.data_dir = data_dir
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
        self.subject_cache = {}

        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(self.device)
        self.model = None
        self._reset_model()

    @staticmethod
    def get_seq_len(n_channels: int = 30, n_times: int = 250, emb_size: int = 24) -> int:
        dummy = torch.zeros(1, 1, n_channels, n_times)
        pe = PatchEmbedding(emb_size, n_channels)
        with torch.no_grad():
            out = pe(dummy)
        return out.shape[1]

    def augment(self, x: Tensor) -> Tensor:
        x = x + torch.randn_like(x) * self.aug_noise_std
        shift = torch.randint(-self.aug_shift, self.aug_shift + 1, (1,), device=x.device).item()
        x = torch.roll(x, shift, dims=-1)
        if self.channel_mask_prob > 0 and torch.rand(1, device=x.device).item() < self.channel_mask_prob:
            mask = (torch.rand(x.size(0), 1, x.size(2), 1, device=x.device) > self.channel_drop_prob).float()
            x = x * mask
        return x

    @staticmethod
    def mixup(x: Tensor, y: Tensor, alpha: float = 0.2):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
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
        for _ in range(n_views):
            x_aug = self.augment(x.clone())
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                _, out = self.model(x_aug)
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
                        _, logits = self.model(batch)
            except RuntimeError as exc:
                if self.device.type != "cuda" or "cuDNN algorithm" not in str(exc):
                    raise
                with torch.backends.cudnn.flags(enabled=False):
                    if n_tta and n_tta > 1:
                        logits = self.tta_evaluate(batch, n_views=n_tta)
                    else:
                        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                            _, logits = self.model(batch)
            logits_list.append(logits)
        return torch.cat(logits_list, dim=0)

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
        train_concat = self._apply_normalization(train_base, train_mu, train_std)
        train_concat = np.ascontiguousarray(train_concat[:, np.newaxis], dtype=np.float32)
        label_concat = np.ascontiguousarray(label_concat, dtype=np.int64)

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
            train_subject_ids,
            val_subject_ids,
        )

    def run_subject_cv(
        self,
        subject_ids,
        save_dir,
        n_epochs=120,
        batch_size=128,
        patience=20,
        seed=42,
        val_ratio=0.125,
        n_tta=1,
        num_workers=4,
        start_fold=1,
        end_fold=999,
        common_ckpt_name="raw_finetuned_best.pth",
    ):
        os.makedirs(save_dir, exist_ok=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_splits = list(enumerate(kf.split(np.array(subject_ids)), 1))
        end_fold = min(end_fold, len(fold_splits))

        fold_results = []
        processed_folds = []
        best_overall_val = float("-inf")
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
                train_subjects,
                val_subjects,
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
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(val_data, dtype=torch.float32),
                torch.tensor(val_label, dtype=torch.long),
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

            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=(self.b1, self.b2),
                weight_decay=self.weight_decay,
            )
            scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=n_epochs)
            ema = EMA(self.model, decay=0.999)
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

            best_val_acc = float("-inf")
            patience_counter = 0
            best_save_path = os.path.join(save_dir, f"raw_subjectcv_fold{fold_idx}_best.pth")

            for epoch in range(n_epochs):
                self.model.train()
                train_loss = 0.0
                train_correct = 0

                for imgs, labels in train_loader:
                    imgs = imgs.to(self.device, non_blocking=use_pin_memory)
                    labels = labels.to(self.device, non_blocking=use_pin_memory)
                    imgs = self.augment(imgs)

                    use_mixup = torch.rand(1, device=self.device).item() < self.mixup_prob
                    optimizer.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        if use_mixup:
                            imgs, labels_a, labels_b, lam = ExGAN.mixup(imgs, labels, alpha=self.mixup_alpha)
                            _, outputs = self.model(imgs)
                            loss = lam * self.criterion_cls(outputs, labels_a) + (1 - lam) * self.criterion_cls(outputs, labels_b)
                        else:
                            _, outputs = self.model(imgs)
                            loss = self.criterion_cls(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    ema.update()

                    train_loss += loss.item() * len(imgs)
                    if use_mixup:
                        pred = outputs.argmax(1)
                        mix_acc = lam * (pred == labels_a).float() + (1 - lam) * (pred == labels_b).float()
                        train_correct += mix_acc.sum().item()
                    else:
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
                        v_imgs = v_imgs.to(self.device, non_blocking=use_pin_memory)
                        v_labels = v_labels.to(self.device, non_blocking=use_pin_memory)
                        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                            _, v_outputs = self.model(v_imgs)
                            v_loss = self.criterion_cls(v_outputs, v_labels)
                        val_loss += v_loss.item() * len(v_imgs)
                        val_correct += (v_outputs.argmax(1) == v_labels).sum().item()
                ema.restore()

                avg_val_loss = val_loss / len(val_label)
                avg_val_acc = val_correct / len(val_label)

                print(
                    f"Epoch {epoch + 1:3d}/{n_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}"
                )

                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    ema.apply_shadow()
                    torch.save(self._cpu_state_dict(self.model), best_save_path)
                    ema.restore()
                    patience_counter = 0
                    print(f"new best: save to: {best_save_path}")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping after {patience} epochs without val improvement.")
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

            if best_val_acc > best_overall_val:
                best_overall_val = best_val_acc
                torch.save(best_state, common_ckpt_path)

            print(
                f"Fold {fold_idx} Test Acc: {test_acc * 100:.2f}% "
                f"(best val: {best_val_acc * 100:.2f}%)"
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

        with open(os.path.join(save_dir, "RAW_CV_RESULTS.txt"), "w", encoding="utf-8") as f:
            f.write("EEG-Conformer-Raw 5-Fold CV Results\n")
            f.write(f"Folds run: {processed_folds}\n")
            f.write(f"Fold accuracies: {fold_results}\n")
            f.write(f"Mean: {mean_acc * 100:.2f}% +- {std_acc * 100:.2f}%\n")
            f.write(f"Best shared checkpoint: {common_ckpt_path}\n")

        return fold_results


def parse_args():
    parser = argparse.ArgumentParser(description="EEG-Conformer-Raw subject-level 5-fold CV training.")
    parser.add_argument("--data_dir", type=str, default="./EEG-Conformer/data/processed_normal/")
    parser.add_argument("--save_dir", type=str, default="./EEG-Conformer/last_params/")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--emb_size", type=int, default=24)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_seed", type=int, default=-1)
    parser.add_argument("--val_ratio", type=float, default=0.125)
    parser.add_argument("--n_tta", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_denoise", action="store_true", default=False)
    parser.add_argument("--disable_amp", action="store_true", default=False)
    parser.add_argument("--aug_noise_std", type=float, default=0.01)
    parser.add_argument("--aug_shift", type=int, default=8)
    parser.add_argument("--channel_mask_prob", type=float, default=0.0)
    parser.add_argument("--channel_drop_prob", type=float, default=0.0)
    parser.add_argument("--mixup_prob", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--end_fold", type=int, default=1)
    parser.add_argument("--common_ckpt_name", type=str, default="raw_finetuned_best.pth")
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

    seq_len = ExGAN.get_seq_len(n_channels=30, n_times=250, emb_size=args.emb_size)
    print(f"[Info] seq_len = {seq_len}")

    start_time = datetime.datetime.now()
    model_seed = args.model_seed if args.model_seed >= 0 else args.seed

    trainer = ExGAN(
        data_dir=args.data_dir,
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
    )
    trainer.run_subject_cv(
        subject_ids=subject_ids,
        save_dir=args.save_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
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
