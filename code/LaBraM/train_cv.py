"""
Modified train_de_cv.py: add --extra_data_dir support for LaBraM + depression.
Extra subjects are added to every fold's training set only.
"""
import os, sys, numpy as np, torch, torch.backends.cudnn as cudnn
from pathlib import Path
from collections import OrderedDict
from sklearn.model_selection import KFold

from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy
import modeling_finetune
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from engine_for_finetuning import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from dataset_de_v2 import CompetitionDEDataset


class TeeLogger:
    def __init__(self, p):
        self.t = sys.stdout; self.f = open(p, 'a', encoding='utf-8')
    def write(self, m): self.t.write(m); self.f.write(m); self.f.flush()
    def flush(self): self.t.flush(); self.f.flush()


def build_datasets(data_dir, train_files, test_files, extra_dirs, extra_files_list, window_size, stride):
    train_ds = CompetitionDEDataset(data_dir, 'train', train_files, window_size=window_size, stride=stride)
    test_ds = CompetitionDEDataset(data_dir, 'test', test_files, window_size=window_size, stride=stride)

    extra_datasets = []
    for edir, efiles in zip(extra_dirs, extra_files_list):
        eds = CompetitionDEDataset(edir, 'extra', efiles, window_size=window_size, stride=stride)
        extra_datasets.append(eds)

    if extra_datasets:
        train_ds = torch.utils.data.ConcatDataset([train_ds] + extra_datasets)
    return train_ds, test_ds, train_ds.datasets[0].ch_names if extra_datasets else train_ds.ch_names


def run_fold(fold_idx, data_dir, train_files, test_files, extra_dirs, extra_files_list, args, device):
    print(f"\n{'='*50}\nFold {fold_idx+1}/5: {len(train_files)} train / {len(test_files)} test subjects")
    n_extra = sum(len(ef) for ef in extra_files_list)
    if n_extra > 0:
        print(f"  + {n_extra} extra training-only subjects from {len(extra_dirs)} dir(s)")
    print(f"{'='*50}")

    ds_train, ds_test, ch_names = build_datasets(data_dir, train_files, test_files, extra_dirs, extra_files_list, args.de_window_size, args.de_stride)
    dl_train = torch.utils.data.DataLoader(ds_train, sampler=torch.utils.data.RandomSampler(ds_train), batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_test = torch.utils.data.DataLoader(ds_test, sampler=torch.utils.data.SequentialSampler(ds_test), batch_size=int(1.5*args.batch_size), num_workers=args.num_workers, pin_memory=True)

    model = create_model(args.model, pretrained=False, num_classes=2, drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0, use_mean_pooling=True, init_scale=0.001, use_rel_pos_bias=True, use_abs_pos_emb=True, init_values=0.1, qkv_bias=True)
    patch_size = model.patch_size

    ckpt = torch.load(args.finetune, map_location='cpu', weights_only=False)
    ckpt_m = None
    for mk in 'model|module'.split('|'):
        if mk in ckpt: ckpt_m = ckpt[mk]; break
    if ckpt_m is None: ckpt_m = ckpt
    nd = OrderedDict()
    for k in list(ckpt_m.keys()):
        if k.startswith('student.'): nd[k[8:]] = ckpt_m[k]
    if nd: ckpt_m = nd
    sd = model.state_dict()
    for k in ['head.weight','head.bias']:
        if k in ckpt_m and ckpt_m[k].shape != sd[k].shape: del ckpt_m[k]
    for k in list(ckpt_m.keys()):
        if "relative_position_index" in k: ckpt_m.pop(k)
    utils.load_state_dict(model, ckpt_m, prefix='')
    model.to(device)

    n_steps = len(ds_train) // (args.batch_size * utils.get_world_size())
    num_layers = model.get_num_layers()
    assigner = LayerDecayValueAssigner(list(0.9**(num_layers+1-i) for i in range(num_layers+2)))
    opt = create_optimizer(args, model, skip_list=model.no_weight_decay(), get_num_layer=assigner.get_layer_id, get_layer_scale=assigner.get_scale)
    scaler = NativeScaler()
    lr_vals = utils.cosine_scheduler(args.lr, args.min_lr, args.epochs, n_steps, warmup_epochs=args.warmup_epochs)
    wd_end = args.weight_decay if args.weight_decay_end is None else args.weight_decay_end
    wd_vals = utils.cosine_scheduler(args.weight_decay, wd_end, args.epochs, n_steps)
    crit = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    metrics = ["accuracy","balanced_accuracy","cohen_kappa","f1_weighted"]

    best_test = 0.0
    for epoch in range(args.epochs):
        train_one_epoch(model, crit, dl_train, opt, device, epoch, scaler, None, None, start_steps=epoch*n_steps, lr_schedule_values=lr_vals, wd_schedule_values=wd_vals, num_training_steps_per_epoch=n_steps, update_freq=1, ch_names=ch_names, is_binary=False, patch_size=patch_size, scale=1.0)
        tts = evaluate(dl_test, model, device, 'Test:', ch_names=ch_names, metrics=metrics, is_binary=False, patch_size=patch_size, scale=1.0)
        if tts["accuracy"] > best_test: best_test = tts["accuracy"]
        if (epoch+1) % 5 == 0:
            print(f"  Fold {fold_idx+1} Epoch {epoch+1}/{args.epochs} | Best Test: {best_test:.2f}%")

    print(f"Fold {fold_idx+1} Final Best Test: {best_test:.2f}%")
    return best_test


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='./de_data')
    p.add_argument('--extra_data_dir', nargs='*', default=[], help='extra training-only dirs (e.g. depression)')
    p.add_argument('--output_dir', default='./output_de_cv')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch_size', default=32, type=int)
    p.add_argument('--epochs', default=20, type=int)
    p.add_argument('--de_window_size', default=40, type=int)
    p.add_argument('--de_stride', default=5, type=int)
    p.add_argument('--start_fold', default=0, type=int)
    p.add_argument('--end_fold', default=4, type=int)
    p.add_argument('--model', default='labram_base_patch200_200')
    p.add_argument('--finetune', default='./checkpoints/labram-base.pth')
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--world_size', default=1, type=int); p.add_argument('--local_rank', default=-1, type=int)
    p.add_argument('--dist_url', default='env://'); p.add_argument('--dist_on_itp', action='store_true')
    p.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    p.add_argument('--model_ema', action='store_true', default=False)
    p.add_argument('--model_ema_decay', type=float, default=0.9999)
    p.add_argument('--model_ema_force_cpu', action='store_true', default=False)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--weight_decay_end', type=float, default=None)
    p.add_argument('--opt', default='adamw'); p.add_argument('--opt_eps', default=1e-8, type=float)
    p.add_argument('--opt_betas', default=None, type=float, nargs='+')
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--lr', type=float, default=5e-4); p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--warmup_lr', type=float, default=1e-6); p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--warmup_steps', type=int, default=-1); p.add_argument('--smoothing', type=float, default=0.1)
    args = p.parse_args()

    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = True

    de_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('_X.npy')])
    extra_files_list = []
    for ed in args.extra_data_dir:
        ef = sorted([f for f in os.listdir(ed) if f.endswith('_X.npy')])
        extra_files_list.append(ef)
        print(f'Extra data dir: {ed} ({len(ef)} subjects)')
    print(f'Main data dir: {args.data_dir} ({len(de_files)} subjects)')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    de_files = np.array(de_files)

    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(de_files)):
        if fold_idx < args.start_fold or fold_idx > args.end_fold:
            continue
        train_files = de_files[train_idx].tolist()
        test_files = de_files[test_idx].tolist()
        best_test = run_fold(fold_idx, args.data_dir, train_files, test_files, args.extra_data_dir, extra_files_list, args, device)
        results.append(best_test)
        print(f"\nRunning results: {[f'{r:.2f}%' for r in results]} | Mean: {np.mean(results):.2f}% | Std: {np.std(results):.2f}%")

    print(f"\n{'='*60}")
    print(f"5-Fold CV Complete")
    print(f"Individual folds: {[f'{r:.2f}%' for r in results]}")
    print(f"Mean Test Acc: {np.mean(results):.2f}% +- {np.std(results):.2f}%")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "CV_RESULTS.txt"), 'w') as f:
        f.write(f"5-Fold CV Results\n")
        f.write(f"Fold results: {results}\n")
        f.write(f"Mean: {np.mean(results):.2f}% +- {np.std(results):.2f}%\n")


if __name__ == '__main__':
    import sys as _sys
    _out_dir = './output_de_cv'
    for _i, _a in enumerate(_sys.argv):
        if _a == '--output_dir' and _i + 1 < len(_sys.argv):
            _out_dir = _sys.argv[_i + 1]
            break
    Path(_out_dir).mkdir(parents=True, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(_out_dir, 'training_log.txt'))
    main()
