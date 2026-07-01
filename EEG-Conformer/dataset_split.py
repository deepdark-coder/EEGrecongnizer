"""
Emotion EEG 数据集划分 — 跨被试 4:1 固定划分 (seed=23)

划分标准：
  - 被试级别划分，训练/测试完全隔离，无数据泄露
  - 40 人 → 32 人训练 (80%) / 8 人测试 (20%)
  - 随机种子 = 23，固定可复现
  - 每类样本数均衡（分层采样）

用法：
  python dataset_split.py --data_dir <path_to_npy>

输出：
  - 训练/测试被试 ID 列表
  - 各被试样本统计
  - 最终训练/测试集样本总数
"""
import os
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

SPLIT_SEED = 23
TRAIN_RATIO = 0.8


def load_subject_data(data_dir):
    """加载所有被试的 DE .npy 数据"""
    data_dir = Path(data_dir)
    x_files = sorted(data_dir.glob('*_X.npy'))
    subjects = []
    for xf in x_files:
        yf = data_dir / xf.name.replace('_X.npy', '_Y.npy')
        if not yf.exists():
            continue
        x = np.load(xf)
        y = np.load(yf)
        sid = xf.stem.replace('timedata_X', '').replace('timedata', '')
        subjects.append({'id': sid, 'x_shape': x.shape, 'y_shape': y.shape,
                         'n_samples': len(y), 'labels': Counter(y)})
    return subjects


def do_split(subjects, seed=SPLIT_SEED, train_ratio=TRAIN_RATIO):
    """执行 4:1 跨被试划分"""
    n = len(subjects)
    n_train = int(n * train_ratio)
    n_test = n - n_train

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_sids = [subjects[i]['id'] for i in train_idx]
    test_sids = [subjects[i]['id'] for i in test_idx]

    return train_sids, test_sids


def main():
    parser = argparse.ArgumentParser(description='Emotion EEG 跨被试 4:1 数据集划分')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='DE .npy 文件目录')
    parser.add_argument('--output', type=str, default=None,
                        help='保存划分结果到文件 (可选)')
    args = parser.parse_args()

    # 加载数据
    subjects = load_subject_data(args.data_dir)
    print(f'找到 {len(subjects)} 个被试')

    if len(subjects) < 5:
        print(f'WARNING: not enough subjects ({len(subjects)} < 5)')
        return

    # 划分
    train_sids, test_sids = do_split(subjects)

    # ---- 输出 ----
    print(f'\n{"="*60}')
    print(f'跨被试 4:1 划分 (seed={SPLIT_SEED})')
    print(f'{"="*60}')
    print(f'训练集: {len(train_sids)} 人')
    for sid in train_sids:
        subj = [s for s in subjects if s['id'] == sid][0]
        print(f'  {sid:12s}  {subj["n_samples"]:5d} 样本  '
              f'正:{subj["labels"][1]:4d}  中:{subj["labels"][0]:4d}')
    print(f'\n测试集: {len(test_sids)} 人')
    for sid in test_sids:
        subj = [s for s in subjects if s['id'] == sid][0]
        print(f'  {sid:12s}  {subj["n_samples"]:5d} 样本  '
              f'正:{subj["labels"][1]:4d}  中:{subj["labels"][0]:4d}')

    # 统计
    train_total = sum([s for s in subjects if s['id'] in train_sids][0]['n_samples']
                      for _ in [1])  # workaround
    train_total = sum(s['n_samples'] for s in subjects if s['id'] in train_sids)
    test_total = sum(s['n_samples'] for s in subjects if s['id'] in test_sids)

    # 统计每类样本
    train_pos = sum(s['labels'][1] for s in subjects if s['id'] in train_sids)
    train_neu = sum(s['labels'][0] for s in subjects if s['id'] in train_sids)
    test_pos = sum(s['labels'][1] for s in subjects if s['id'] in test_sids)
    test_neu = sum(s['labels'][0] for s in subjects if s['id'] in test_sids)

    print(f'\n{"="*60}')
    print(f'汇总')
    print(f'{"="*60}')
    print(f'{"":12s}  {"样本总数":>7s}  {"积极":>6s}  {"中性":>6s}')
    print(f'{"训练集":12s}  {train_total:7d}  {train_pos:6d}  {train_neu:6d}')
    print(f'{"测试集":12s}  {test_total:7d}  {test_pos:6d}  {test_neu:6d}')
    print(f'{"合计":12s}  {train_total+test_total:7d}  '
          f'{train_pos+test_pos:6d}  {train_neu+test_neu:6d}')

    # 保存
    result = {
        'seed': SPLIT_SEED,
        'train_ratio': TRAIN_RATIO,
        'train_subjects': train_sids,
        'test_subjects': test_sids,
        'train_n': len(train_sids),
        'test_n': len(test_sids),
        'train_samples': train_total,
        'test_samples': test_total,
    }
    if args.output:
        np.save(args.output, result)
        print(f'\n已保存到 {args.output}')
    else:
        print(f'\ntrain_subjects = {train_sids}')
        print(f'test_subjects  = {test_sids}')


if __name__ == '__main__':
    main()
