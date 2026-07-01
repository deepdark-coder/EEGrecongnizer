# EEGRecognizer

一个面向 EEG 二分类任务的实验仓库，包含多条图网络、预训练 Transformer、卷积 Transformer 与简单融合推理方案。仓库目前的重点是：

- 基于 DE 特征的 `EdgeConv`、`LaBraM` 等路线
- 基于时序 EEG 片段的 `EEG-Conformer` 与 `conformer_raw` 路线
- 面向实际使用的轻量融合推理脚本 `infer/simple_fusion.py`

这份 README 用来统一说明整个仓库的结构、数据格式和推荐使用方式。

## 目录结构

```text
EEGrecongnizer/
├─ code/                  # 多种 DE / 图模型实验代码
├─ EEG-Conformer/         # Conformer 系列模型与 5 折训练脚本
├─ infer/                 # 简化后的多模型融合推理
├─ data/                  # 数据目录（按你的实际情况放置）
├─ params/                # 训练参数或中间配置
├─ results/               # 实验结果输出
├─ output_de_cv/          # LaBraM 等 CV 输出
└─ README.md              # 当前总说明
```

## 仓库包含的主要模型

### `code/` 下的 DE / 图模型

- `Chebyshev_GCN`
  经典图卷积基线，固定图结构，适合做稳定对照。
- `EdgeConv`
  当前仓库里最完整的图模型路线，支持动态 k-NN、Band SE、DANN、MixUp、TTA 等扩展。
- `GAT`
  图注意力路线，整体训练骨架与 `EdgeConv` 类似。
- `MoGE`
  图专家混合模型，可看成 `EdgeConv` 的结构变体。
- `RawWaveform`
  直接基于原始波形的图模型实验。
- `LaBraM`
  基于预训练 EEG Transformer 的微调方案，需要预训练权重 `labram-base.pth`。
- `preprocessing`
  数据清洗、PSD / DE 特征提取等预处理脚本。

### `EEG-Conformer/`

- `conformer.py`
  基于时序 EEG 片段的 Conformer 训练脚本，当前已整理为被试级 5 折交叉验证流程。
- `conformer_raw.py`
  原始波形路线，对应另一条 5 折训练脚本。
- `README.md`
  上游 EEG-Conformer 项目的简要介绍。

### `infer/`

- `simple_fusion.py`
  简化后的晚期融合脚本，直接调用 `EdgeConv`、`LaBraM`、`EEG-Conformer` 的模型结构做推理，并在 `(subject_id, label)` 层面做概率平均。
- `README.md`
  融合脚本的单独说明。

## 数据格式

仓库目前主要涉及两类输入。

### 1. DE 特征 `.npy`

供 `EdgeConv`、`LaBraM`、`GAT`、`MoGE` 等模型使用。

```text
*_X.npy: (N, 30, 5)
*_Y.npy: (N,)
```
可理解为：

- `N`：时间片或滑窗样本数
- `30`：EEG 通道数
- `5`：频带特征数



### 2. 时序 EEG `.mat`

供 `EEG-Conformer` 路线使用。

当前脚本默认读取类似：

```text
HC1001_1s.mat
HC1002_1s.mat
...
```

单个样本通常按 1 秒片段处理，对应大致形状：

- `30` 通道
- `250` 采样点

## 推荐环境

建议使用独立虚拟环境。

基础依赖：

- Python 3.10+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- einops


## 各模块如何使用

### 1.  EdgeConv

进入目录后执行：

```bash
cd code/EdgeConv
python train.py --data_dir /path/to/processed_normal --gpu 0 --window_size 40 --stride 1 --k_neighbors 20 --model edgeconv
```

### 2. LaBraM

先确保存在预训练权重：

```text
code/LaBraM/checkpoints/labram-base.pth
```

执行示例：

```bash
cd code/LaBraM
python train_cv.py --data_dir /path/to/processed_normal --finetune ./checkpoints/labram-base.pth
```


### 3. 跑 `EEG-Conformer`

工作模式有两种
- `baseline`
- `full`

分别运行：

```bash
python EEG-Conformer/conformer.py --training_mode baseline --start_fold 1 --end_fold 5
```

```bash
python EEG-Conformer/conformer.py --training_mode full --start_fold 1 --end_fold 5
```


## 融合推理

推荐使用简化版融合入口：

```bash
python infer/simple_fusion.py \
  --de_data_dir processed_normal \
  --edge_ckpt path/to/edgeconv.pth \
  --labram_ckpt path/to/labram.pth \
  --eeg_ckpt path/to/conformer.pth
```

追加抑郁症患者测试目录：

```bash
python infer/simple_fusion.py \
  --de_data_dir processed_normal \
  --mat_data_dir path/to/data.m
  --extra_data_dir path/to/extra_data.m
  --edge_ckpt path/to/edgeconv.pth \
  --labram_ckpt path/to/labram.pth \
  --eeg_ckpt path/to/conformer.pth
```

输出默认写到：

```text
infer/output/fused_subject_label_scores.csv
```


