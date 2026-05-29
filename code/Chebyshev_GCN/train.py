"""
DGCNN Chebyshev GCN — 5-fold Cross-Validation Training Script.

Usage:
  python train.py --data_dir <path/to/npy> --gpu 0
  python train.py --data_dir <path> --gpu 0 --window_size 40 --stride 1 --k_adj 25 --extra_data_dir <depression_path>
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

from model import DGCNN
from utils import eegDataset, load_subject_data, build_windows, add_asymmetry_channels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True,
                   help='path to main DE npy files for CV split')
    p.add_argument('--extra_data_dir', type=str, nargs='*', default=[],
                   help='paths to extra training-only DE npy dirs (e.g. depression)')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--window_size', type=int, default=1, help='temporal window splicing size')
    p.add_argument('--stride', type=int, default=1, help='sliding stride')
    p.add_argument('--k_adj', type=int, default=10, help='Chebyshev polynomial order')
    p.add_argument('--num_out', type=int, default=64, help='GCN hidden dim')
    p.add_argument('--num_gcn_layers', type=int, default=2, help='stacked Chebynet layers')
    p.add_argument('--gcn_dropout', type=float, default=0.2, help='dropout after Chebynet')
    p.add_argument('--use_asymmetry', action='store_true', default=False,
                   help='append left-right asymmetry differential channels (30->42 ch)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--val_ratio', type=float, default=0.125, help='val subjects ratio from train set')
    p.add_argument('--seed', type=int, default=42, help='random seed for KFold split')
    p.add_argument('--model_seed', type=int, default=-1,
                   help='additional seed for model init (-1 = same as KFold seed)')
    p.add_argument('--start_fold', type=int, default=1)
    p.add_argument('--end_fold', type=int, default=999)
    p.add_argument('--output_dir', type=str, default='./results')
    p.add_argument('--label_smoothing', type=float, default=0.0,
                   help='Label Smoothing factor (0=disabled, 0.1 recommended)')
    return p.parse_args()


def train_epoch(model, loader, criterion, optimizer, device, label_smoothing=0.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_smooth = label_smoothing > 0
    for batch in loader:
        X, Y = batch[0], batch[1]
        X, Y = X.float().to(device), Y.long().to(device)
        optimizer.zero_grad()
        output = model(X)
        if use_smooth:
            loss = _label_smoothing_ce(output, Y, label_smoothing)
        else:
            loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == Y).sum().item()
        total += Y.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_data=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_logits, all_labels = [], []
    for batch in loader:
        X, Y = batch[0], batch[1]
        X, Y = X.float().to(device), Y.long().to(device)
        output = model(X)
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


def _label_smoothing_ce(logits, targets, smoothing):
    n_classes = logits.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(true_dist * log_probs).sum(dim=-1).mean()


def prepare_subject_data(subjects, subject_ids, window_size, stride, use_asymmetry=False):
    all_x, all_y, all_subj = [], [], []
    for i, (x, y) in enumerate(subjects):
        x = x.astype(np.float64)
        x = zscore(x, axis=0)
        x, y = build_windows(x, y, window_size, stride)
        if use_asymmetry:
            x = add_asymmetry_channels(x)
        x = x.astype(np.float32)
        y = y.astype(np.int64)
        all_x.append(torch.tensor(x))
        all_y.append(torch.tensor(y))
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

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_splits = list(enumerate(kf.split(np.arange(len(main_subjects))), 1))
    n_folds = len(fold_splits)

    end_fold = min(args.end_fold, n_folds)
    acc_all = []

    for fold_idx, (train_val_idx, test_idx) in fold_splits:
        if fold_idx < args.start_fold or fold_idx > end_fold:
            continue

        print(f'\n{"="*60}\nFold {fold_idx}/{n_folds}\n{"="*60}')

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

        print(f'Train: {len(train_y)} samples, Val: {len(val_y)}, Test: {len(test_y)}')

        train_loader = DataLoader(eegDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(eegDataset(val_x, val_y), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(eegDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

        model_seed = args.model_seed if args.model_seed >= 0 else args.seed
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)

        model = DGCNN(in_features=in_features, num_nodes=num_nodes, k_adj=args.k_adj,
                      num_out=args.num_out, nclass=2,
                      gcn_dropout=args.gcn_dropout,
                      num_gcn_layers=args.num_gcn_layers).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'Model: DGCNN Chebyshev ({n_params:,} parameters)')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args.label_smoothing)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f'  Epoch {epoch:3d} | Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | '
                      f'Val Loss {val_loss:.4f} Acc {val_acc*100:.2f}%')

            if patience_counter >= args.patience:
                print(f'  Early stopping at epoch {epoch}')
                break

        model.load_state_dict(best_state)
        test_loss, test_acc, test_logits, test_labels = evaluate(
            model, test_loader, criterion, device, return_data=True)
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
