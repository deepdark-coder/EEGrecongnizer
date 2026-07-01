# infer

这个目录是一个更简化的联合推理方案，和原来的 `conbine/` 并存，互不影响。

这里只保留两个文件：

- `simple_fusion.py`
- `README.md`

## 简化思路

不再做复杂的逐窗口对齐，也不再先导出三份中间 CSV 再二次融合。

这里改成更稳定的晚融合方法：

1. 每个模型按自己的原始输入方式独立跑推理
2. 对同一个 `(subject_id, label)` 的所有片段概率取平均
3. 再对三个模型的平均概率做加权平均

这样做的好处是：

- 不需要很多辅助脚本
- 不需要严格对齐三种不同的时间窗
- 不修改原始模型源码
- 逻辑更直白，后续更容易自己改

## 这个脚本会直接调用哪些原始代码

- `code/EdgeConv/model.py`
- `code/EdgeConv/spatial_prior.py`
- `code/LaBraM/modeling_finetune.py`
- `EEG-Conformer/conformer.py`

也就是说，新脚本只是“外部包装调用”，没有改你原来的训练和模型文件。

## 依赖

最少需要：

- `torch`
- `numpy`

如果你启用 EEG-Conformer，还需要：

- `scipy`

如果你启用 LaBraM，还需要：

- `timm`

## 最常用的运行方式

### 1. EdgeConv + EEG-Conformer

```bash
python infer/simple_fusion.py \
  --de_data_dir processed_normal \
  --edge_ckpt path/to/edgeconv.pth \
  --eeg_ckpt EEG-Conformer/last_params/better_D2_H4_S40_best1.pth
```

### 2. 三个模型一起融合

```bash
python infer/simple_fusion.py \
  --de_data_dir processed_normal \
  --edge_ckpt path/to/edgeconv.pth \
  --labram_ckpt path/to/labram.pth \
  --eeg_ckpt EEG-Conformer/last_params/better_D2_H4_S40_best1.pth
```

### 3. 手动调整权重

```bash
python infer/simple_fusion.py \
  --de_data_dir processed_normal \
  --edge_ckpt path/to/edgeconv.pth \
  --labram_ckpt path/to/labram.pth \
  --eeg_ckpt EEG-Conformer/last_params/better_D2_H4_S40_best1.pth \
  --edge_weight 1.0 \
  --labram_weight 1.2 \
  --eeg_weight 1.0
```

### 4. 追加 extra 测试数据目录

如果你有新的测试数据放在别的目录，可以直接追加：

```bash
python infer/simple_fusion.py \
  --de_data_dir processed_normal \
  --mat_data_dir EEG-Conformer/data/processed_normal \
  --extra_data_dir new_testset_dir another_testset_dir \
  --edge_ckpt path/to/edgeconv.pth \
  --labram_ckpt path/to/labram.pth \
  --eeg_ckpt EEG-Conformer/last_params/better_D2_H4_S40_best1.pth
```

这个参数会自动识别：

- 包含 `*_X.npy` / `*_Y.npy` 的目录，会并入 EdgeConv 和 LaBraM
- 包含 `*.mat` 的目录，会并入 EEG-Conformer

注意：

- 如果不同目录里有同名 `subject_id`，脚本会直接报错，避免静默覆盖
- 最终融合仍然要求三个模型在 `(subject_id, label)` 上能对齐

## 输出

默认输出到：

`infer/output/fused_subject_label_scores.csv`

表头很简单：

- `subject_id`
- `label`
- `edgeconv`
- `labram`
- `eeg_conformer`
- `fused_prob_1`
- `fused_pred`

其中：

- `edgeconv / labram / eeg_conformer` 是每个模型对该 `(subject_id, label)` 的平均正类概率
- `fused_prob_1` 是最后加权平均后的概率
- `fused_pred` 是最终预测类别

## 说明

这个版本是“为了更容易跑通和维护”而做的简化版，不是最复杂也不是最花哨的融合方案。

如果你后面确认这个简化版流程稳定，再决定要不要回到更细粒度的 chunk-level 融合，会更稳一些。
