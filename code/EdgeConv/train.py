"""
EdgeConv DGCNN — 5-fold Cross-Validation Training Script.

Features: dynamic k-NN graph, spatial prior, Band SE, DANN adversarial adaptation,
          SupCon contrastive learning, MixUp augmentation, Focal Loss, Label Smoothing,
          TTA (test-time augmentation), AdaBN, hemispheric asymmetry channels, depression extra data.

Best configs:
  # Baseline (82.76% SOTA)
  python train.py --data_dir <path> --gpu 0 --window_size 40 --stride 1 --k_neighbors 20 --model edgeconv

  # With Depression extra data (83.92% NEW SOTA)
  python train.py --data_dir <path> --gpu 0 --window_size 40 --stride 1 --k_neighbors 20 \
      --model edgeconv --extra_data_dir <depression_path>

  # 3-seed ensemble
  python train.py --data_dir <path> --gpu 0 --model edgeconv --seed 42 --model_seed 1
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from scipy.stats import zscore

from model import EdgeDGCNN, EdgeDGCNN_DANN, adapt_bn
from utils import eegDataset, load_subject_data, build_windows, add_asymmetry_channels
from spatial_prior import get_spatial_dist


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--extra_data_dir', type=str, nargs='*', default=[])
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--window_size', type=int, default=40)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--k_neighbors', type=int, default=20, help='k nearest neighbors for EdgeConv')
    p.add_argument('--model', type=str, default='edgeconv',
                   choices=['edgeconv', 'edgeconv_dann'],
                   help='model architecture')
    p.add_argument('--lambda_dann', type=float, default=0.1, help='DANN adversarial loss weight')
    p.add_argument('--dann_domain', action='store_true', default=False,
                   help='DANN discriminates domain (normal vs depression) instead of subject ID')
    p.add_argument('--mixup_alpha', type=float, default=0.0,
                   help='within-subject MixUp alpha (0=disabled, 0.2 recommended)')
    p.add_argument('--supcon_weight', type=float, default=0.0,
                   help='Supervised Contrastive loss weight (0=disabled, 0.1 recommended)')
    p.add_argument('--supcon_temp', type=float, default=0.07, help='SupCon temperature')
    p.add_argument('--tta', action='store_true', default=False, help='enable test-time augmentation')
    p.add_argument('--tta_steps', type=int, default=10)
    p.add_argument('--focal_gamma', type=float, default=0.0, help='Focal Loss gamma (0=disabled, 2.0 recommended)')
    p.add_argument('--label_smoothing', type=float, default=0.0, help='Label Smoothing (0=disabled, 0.1 recommended)')
    p.add_argument('--use_adabn', action='store_true', default=False, help='enable AdaBN test-time adaptation')
    p.add_argument('--use_asymmetry', action='store_true', default=False,
                   help='append left-right asymmetry differential channels (30->42 ch)')
    p.add_argument('--use_band_se', action='store_true', default=False,
                   help='enable band-level SE attention (5 freq bands)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--val_ratio', type=float, default=0.125)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model_seed', type=int, default=-1)
    p.add_argument('--start_fold', type=int, default=1)
    p.add_argument('--end_fold', type=int, default=999)
    p.add_argument('--output_dir', type=str, default='./results')
    return p.parse_args()


def supcon_loss(features, labels, tau=0.07):
    """Supervised contrastive loss: pull same-class samples together in embedding space."""
    B = features.size(0)
    sim = torch.matmul(features, features.T) / tau
    mask = torch.eye(B, device=features.device).bool()
    sim = sim.masked_fill(mask, float('-inf'))
    pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    pos_mask = pos_mask.masked_fill(mask, False)
    exp_sim = sim.exp()
    denom = exp_sim.sum(dim=1, keepdim=True)
    pos_num = pos_mask.sum(dim=1).float().clamp(min=1)
    pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
    loss = -(pos_sum / denom.squeeze() + 1e-8).log().sum() / B
    return loss


def focal_loss(logits, targets, gamma=2.0, label_smoothing=0.0):
    """Focal Loss with optional label smoothing."""
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    focal_weight = (1 - pt) ** gamma
    if label_smoothing > 0:
        ce_smooth = F.cross_entropy(logits, targets, label_smoothing=label_smoothing, reduction='none')
        return (focal_weight * ce_smooth).mean()
    return (focal_weight * ce).mean()


def within_subject_mixup(x, y, s, alpha=0.2):
    """MixUp within same subject: mix pairs of samples from the same subject only."""
    B = x.size(0)
    device = x.device
    idx = torch.arange(B, device=device)
    for subj in torch.unique(s):
        mask = (s == subj).nonzero(as_tuple=True)[0]
        if len(mask) >= 2:
            perm = mask[torch.randperm(len(mask), device=device)]
            idx[mask] = perm
    lam = np.random.beta(alpha, alpha, B)
    lam = torch.tensor(np.maximum(lam, 1 - lam), dtype=x.dtype, device=device)
    x_mix = lam.view(-1, 1, 1) * x + (1 - lam.view(-1, 1, 1)) * x[idx]
    return x_mix, y, y[idx], lam.view(-1, 1)


def train_epoch(model, loader, criterion, optimizer, device, is_dann=False, lambda_=0.0,
                lambda_dann=0.1, mixup_alpha=0.0, supcon_weight=0.0,
                focal_gamma=0.0, label_smoothing=0.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_mixup = mixup_alpha > 0
    use_focal = focal_gamma > 0 or label_smoothing > 0
    use_supcon = supcon_weight > 0

    for batch in loader:
        if is_dann or use_mixup:
            X, Y, S = batch
            X, Y, S = X.float().to(device), Y.long().to(device), S.long().to(device)
        else:
            X, Y = batch[:2]
            X, Y = X.float().to(device), Y.long().to(device)

        if use_mixup:
            X, Y_a, Y_b, lam = within_subject_mixup(X, Y, S, mixup_alpha)

        optimizer.zero_grad()

        if use_supcon:
            logits, proj = model(X)
            cls_loss = focal_loss(logits, Y, focal_gamma, label_smoothing) if use_focal else criterion(logits, Y)
            sup_loss = supcon_loss(proj, Y, tau=0.07)
            loss = cls_loss + supcon_weight * sup_loss
            output = logits
        elif is_dann:
            cls_logits, subj_logits = model(X, lambda_=lambda_)
            cls_loss = criterion(cls_logits, Y)
            subj_loss = criterion(subj_logits, S)
            loss = cls_loss + lambda_dann * subj_loss
            output = cls_logits
        elif use_mixup:
            output = model(X)
            if use_focal:
                loss = (lam * focal_loss(output, Y_a, focal_gamma, label_smoothing) +
                        (1 - lam) * focal_loss(output, Y_b, focal_gamma, label_smoothing)).mean()
            else:
                loss = (lam * criterion(output, Y_a) + (1 - lam) * criterion(output, Y_b)).mean()
        else:
            output = model(X)
            loss = focal_loss(output, Y, focal_gamma, label_smoothing) if use_focal else criterion(output, Y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == Y).sum().item()
        total += Y.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_dann=False, return_data=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_logits, all_labels = [], []
    for batch in loader:
        X, Y = batch[0], batch[1]
        X, Y = X.float().to(device), Y.long().to(device)
        if is_dann:
            output = model(X, lambda_=0)
        else:
            output = model(X)
        if isinstance(output, tuple):
            output = output[0]
        loss = criterion(output, Y)
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == Y).sum().item()
        total += Y.size(0)
        if return_data:
            all_logits.append(output)
            all_labels.append(Y)
    if return_data:
        return total_loss / len(loader), correct / total, torch.cat(all_logits), torch.cat(all_labels)
    return total_loss / len(loader), correct / total


def tta_augment(x, noise_std=0.02, chan_drop=3, scale_range=0.1):
    aug = x.clone()
    aug += torch.randn_like(aug) * noise_std
    B, N, C = aug.shape
    for b in range(B):
        drop_idx = torch.randperm(N)[:chan_drop]
        aug[b, drop_idx, :] = 0
    scale = 1.0 + (torch.rand(B, 1, 1, device=x.device) * 2 - 1) * scale_range
    aug *= scale
    return aug


@torch.no_grad()
def evaluate_tta(model, loader, criterion, device, steps=10, is_dann=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        X, Y = batch[0], batch[1]
        X, Y = X.float().to(device), Y.long().to(device)
        logits_sum = None
        for _ in range(steps):
            X_aug = tta_augment(X)
            out = model(X_aug, lambda_=0) if is_dann else model(X_aug)
            if isinstance(out, tuple):
                out = out[0]
            logits_sum = out if logits_sum is None else logits_sum + out
        output = logits_sum / steps
        loss = criterion(output, Y)
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == Y).sum().item()
        total += Y.size(0)
    return total_loss / len(loader), correct / total


def prepare_subject_data(subjects, subject_ids, window_size, stride, use_asymmetry=False):
    all_subj = []
    for i, (x, y) in enumerate(subjects):
        x = x.astype(np.float64)
        x = zscore(x, axis=0)
        x, y = build_windows(x, y, window_size, stride)
        if use_asymmetry:
            x = add_asymmetry_channels(x)
        x = x.astype(np.float32)
        y = y.astype(np.int64)
        all_subj.append((torch.tensor(x), torch.tensor(y), subject_ids[i]))
    return all_subj


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    main_subjects, main_ids = load_subject_data(args.data_dir)
    print(f'Loaded {len(main_subjects)} main subjects')

    extra_subjects = []
    for extra_dir in args.extra_data_dir:
        es, ei = load_subject_data(extra_dir)
        extra_subjects.extend(es)
    if extra_subjects:
        print(f'Loaded {len(extra_subjects)} extra training-only subjects')

    all_subj = prepare_subject_data(main_subjects, main_ids, args.window_size, args.stride, args.use_asymmetry)
    extra_all_subj = prepare_subject_data(
        extra_subjects, [f'extra_{i}' for i in range(len(extra_subjects))],
        args.window_size, args.stride, args.use_asymmetry) if extra_subjects else []

    num_nodes = 42 if args.use_asymmetry else 30
    in_features = all_subj[0][0].shape[-1]
    num_bands = in_features // args.window_size

    spatial_dist = get_spatial_dist(device=device) if not args.use_asymmetry else None

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_splits = list(enumerate(kf.split(np.arange(len(main_subjects))), 1))

    is_dann = args.model == 'edgeconv_dann'
    end_fold = min(args.end_fold, len(fold_splits))
    acc_all = []

    for fold_idx, (train_val_idx, test_idx) in fold_splits:
        if fold_idx < args.start_fold or fold_idx > end_fold:
            continue

        print(f'\n{"="*60}\nFold {fold_idx}/5\n{"="*60}')

        test_subj = [all_subj[i] for i in test_idx]
        test_x = torch.cat([s[0] for s in test_subj])
        test_y = torch.cat([s[1] for s in test_subj])
        test_ids = [s[2] for s in test_subj]
        print(f'Test subjects: {test_ids}')

        n_train_val = len(train_val_idx)
        n_val = max(1, int(n_train_val * args.val_ratio))
        val_idx = np.random.RandomState(42 + fold_idx).choice(train_val_idx, n_val, replace=False)
        train_idx_fold = np.setdiff1d(train_val_idx, val_idx)

        train_subj = [all_subj[i] for i in train_idx_fold]
        val_subj = [all_subj[i] for i in val_idx]

        train_x_list = [s[0] for s in train_subj] + [s[0] for s in extra_all_subj]
        train_y_list = [s[1] for s in train_subj] + [s[1] for s in extra_all_subj]
        train_x = torch.cat(train_x_list)
        train_y = torch.cat(train_y_list)
        val_x = torch.cat([s[0] for s in val_subj])
        val_y = torch.cat([s[1] for s in val_subj])

        need_subject_ids = is_dann or args.mixup_alpha > 0
        if need_subject_ids:
            if args.dann_domain and is_dann:
                train_s_list = []
                for s in train_subj:
                    train_s_list.append(torch.zeros(len(s[1]), dtype=torch.long))
                for s in extra_all_subj:
                    train_s_list.append(torch.ones(len(s[1]), dtype=torch.long))
                val_s_list = [torch.zeros(len(s[1]), dtype=torch.long) for s in val_subj]
            else:
                train_s_list = []
                for si, s in enumerate(train_subj):
                    train_s_list.append(torch.full((len(s[1]),), si, dtype=torch.long))
                for si in range(len(extra_all_subj)):
                    train_s_list.append(torch.full((len(extra_all_subj[si][1]),),
                                                   len(train_subj) + si, dtype=torch.long))
                val_s_list = [torch.full((len(s[1]),), si, dtype=torch.long) for si, s in enumerate(val_subj)]
            train_s = torch.cat(train_s_list)
            val_s = torch.cat(val_s_list)
            train_loader = DataLoader(eegDataset(train_x, train_y, train_s), batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(eegDataset(val_x, val_y, val_s), batch_size=args.batch_size, shuffle=False)
        else:
            train_loader = DataLoader(eegDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(eegDataset(val_x, val_y), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(eegDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

        print(f'Train: {len(train_y)} samples, Val: {len(val_y)}, Test: {len(test_y)}')

        model_seed = args.model_seed if args.model_seed >= 0 else args.seed
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)

        if args.model == 'edgeconv':
            model = EdgeDGCNN(
                in_features=in_features, num_nodes=num_nodes,
                k=args.k_neighbors, nclass=2, spatial_dist=spatial_dist,
                use_supcon=args.supcon_weight > 0,
                use_band_se=args.use_band_se, n_bands=num_bands).to(device)
        elif args.model == 'edgeconv_dann':
            n_dann_classes = 2 if args.dann_domain else len(main_subjects)
            model = EdgeDGCNN_DANN(
                in_features=in_features, num_nodes=num_nodes,
                k=args.k_neighbors, nclass=2,
                num_subjects=n_dann_classes, spatial_dist=spatial_dist).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f'Model: {args.model} ({n_params:,} parameters)')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            p = epoch / args.epochs
            lambda_ = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p)).item()) - 1.0 if is_dann else 0.0

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
                is_dann=is_dann, lambda_=lambda_, lambda_dann=args.lambda_dann,
                mixup_alpha=args.mixup_alpha, supcon_weight=args.supcon_weight,
                focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, is_dann=is_dann)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f'  Epoch {epoch:3d} | Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | '
                      f'Val Loss {val_loss:.4f} Acc {val_acc*100:.2f}% | LR {scheduler.get_last_lr()[0]:.2e}')

            if patience_counter >= args.patience:
                print(f'  Early stopping at epoch {epoch}')
                break

        model.load_state_dict(best_state)
        if args.use_adabn:
            adapt_bn(model, test_loader, device)

        if args.tta:
            test_loss, test_acc = evaluate_tta(model, test_loader, criterion, device,
                                               steps=args.tta_steps, is_dann=is_dann)
        else:
            test_loss, test_acc, test_logits, test_labels = evaluate(
                model, test_loader, criterion, device, is_dann=is_dann, return_data=True)

        acc_all.append(test_acc)
        os.makedirs(os.path.join(args.output_dir, 'logits'), exist_ok=True)
        np.savez(os.path.join(args.output_dir, 'logits', f'fold{fold_idx}.npz'),
                 logits=test_logits.cpu().numpy(), labels=test_labels.cpu().numpy())
        print(f'Fold {fold_idx} Test Acc: {test_acc*100:.2f}% (best val: {best_val_acc*100:.2f}%)')

    acc_all = np.array(acc_all)
    print(f'\n{"="*60}')
    print(f'CV Results: {[f"{a*100:.2f}" for a in acc_all]}')
    print(f'Mean: {acc_all.mean()*100:.2f}% +- {acc_all.std()*100:.2f}%')
    print(f'{"="*60}')
    np.save(os.path.join(args.output_dir, 'cv_results.npy'), acc_all)


if __name__ == '__main__':
    main()
