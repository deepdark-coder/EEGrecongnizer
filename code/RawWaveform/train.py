"""
Raw Waveform EdgeConv — 5-fold Cross-Validation Training Script.

Requires raw waveform .npy files (extracted via raw_extract.py).
Data shape: (N, 30, 250) — 30 channels × 250 time points.

Usage:
  python train.py --data_dir <path/to/raw_npy> --gpu 0

Performance: 67.85% — hand-crafted DE features are a strong inductive bias
that small CNNs cannot replicate with only 40 subjects.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from scipy.stats import zscore

from model import RawEdgeDGCNN
from utils import eegDataset, load_subject_data, build_windows
from spatial_prior import get_spatial_dist


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True, help='path to raw waveform npy files')
    p.add_argument('--extra_data_dir', type=str, nargs='*', default=[])
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--window_size', type=int, default=1, help='window splicing (raw data typically 400 windows)')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--k_neighbors', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--val_ratio', type=float, default=0.125)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model_seed', type=int, default=-1)
    p.add_argument('--start_fold', type=int, default=1)
    p.add_argument('--end_fold', type=int, default=999)
    p.add_argument('--output_dir', type=str, default='./results')
    return p.parse_args()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        X, Y = batch[0], batch[1]
        X, Y = X.float().to(device), Y.long().to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == Y).sum().item()
        total += Y.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        X, Y = batch[0], batch[1]
        X, Y = X.float().to(device), Y.long().to(device)
        output = model(X)
        loss = criterion(output, Y)
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == Y).sum().item()
        total += Y.size(0)
    return total_loss / len(loader), correct / total


def prepare_subject_data(subjects, subject_ids, window_size, stride):
    all_subj = []
    for i, (x, y) in enumerate(subjects):
        x = x.astype(np.float64)
        x = zscore(x, axis=0)
        x, y = build_windows(x, y, window_size, stride)
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

    all_subj = prepare_subject_data(main_subjects, main_ids, args.window_size, args.stride)
    extra_all_subj = prepare_subject_data(
        extra_subjects, [f'extra_{i}' for i in range(len(extra_subjects))],
        args.window_size, args.stride) if extra_subjects else []

    spatial_dist = get_spatial_dist(device=device)

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_splits = list(enumerate(kf.split(np.arange(len(main_subjects))), 1))

    end_fold = min(args.end_fold, len(fold_splits))
    acc_all = []

    for fold_idx, (train_val_idx, test_idx) in fold_splits:
        if fold_idx < args.start_fold or fold_idx > end_fold:
            continue
        print(f'\n{"="*60}\nFold {fold_idx}/5\n{"="*60}')

        test_subj = [all_subj[i] for i in test_idx]
        test_x = torch.cat([s[0] for s in test_subj])
        test_y = torch.cat([s[1] for s in test_subj])
        print(f'Test subjects: {[s[2] for s in test_subj]}')

        n_val = max(1, int(len(train_val_idx) * args.val_ratio))
        val_idx = np.random.RandomState(42 + fold_idx).choice(train_val_idx, n_val, replace=False)
        train_idx_fold = np.setdiff1d(train_val_idx, val_idx)

        train_subj = [all_subj[i] for i in train_idx_fold]
        val_subj = [all_subj[i] for i in val_idx]

        train_x = torch.cat([s[0] for s in train_subj] + [s[0] for s in extra_all_subj])
        train_y = torch.cat([s[1] for s in train_subj] + [s[1] for s in extra_all_subj])
        val_x = torch.cat([s[0] for s in val_subj])
        val_y = torch.cat([s[1] for s in val_subj])

        train_loader = DataLoader(eegDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(eegDataset(val_x, val_y), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(eegDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

        print(f'Train: {len(train_y)}, Val: {len(val_y)}, Test: {len(test_y)}')

        model_seed = args.model_seed if args.model_seed >= 0 else args.seed
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)

        in_len = train_x.shape[-1]
        model = RawEdgeDGCNN(in_len=in_len, num_nodes=30,
                             k=args.k_neighbors, nclass=2,
                             spatial_dist=spatial_dist).to(device)
        print(f'Model: RawEdgeDGCNN ({sum(p.numel() for p in model.parameters()):,} params, in_len={in_len})')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f'  Epoch {epoch:3d} | Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}%')

            if patience_counter >= args.patience:
                print(f'  Early stopping at epoch {epoch}')
                break

        model.load_state_dict(best_state)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        acc_all.append(test_acc)
        print(f'Fold {fold_idx} Test Acc: {test_acc*100:.2f}% (best val: {best_val_acc*100:.2f}%)')

    acc_all = np.array(acc_all)
    print(f'\n{"="*60}')
    print(f'CV Results: {[f"{a*100:.2f}" for a in acc_all]}')
    print(f'Mean: {acc_all.mean()*100:.2f}% +- {acc_all.std()*100:.2f}%')
    print(f'{"="*60)}')
    np.save(os.path.join(args.output_dir, 'cv_results.npy'), acc_all)


if __name__ == '__main__':
    main()
