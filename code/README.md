# Emotion EEG 情绪识别 — 模型代码库

## 目录结构

```
D:/Emotion/code/
├── 01_Chebyshev_GCN/     # Chebyshev 谱图卷积 DGCNN (~80.11%)
├── 02_EdgeConv/          # 动态 k-NN EdgeConv DGCNN (82.76% SOTA / 83.92% +Depression)
├── 03_GAT/               # 图注意力网络 GAT (80.35% / 81.79% +Depression)
├── 04_MoGE/              # 混合图专家 MoGE + EdgeConv (81.05%)
├── 05_RawWaveform/       # 端到端原始波形 EdgeConv (67.85%)
├── 06_LaBraM/            # 预训练 ViT for EEG (83.89%)
└── preprocessing/        # 数据预处理工具
```

## 各路线概览

### 01_Chebyshev_GCN — 谱图卷积
- 模型：学到的邻接矩阵 A → 对称归一化拉普拉斯 → K 阶 Chebyshev 多项式滤波
- 最佳：80.11% (stride=1, K=25, 64d, ws=40)
- 特点：固定图结构，小样本稳定但表达能力受限

### 02_EdgeConv — 动态图卷积 (推荐)
- 模型：每样本动态 k-NN 图 → 边卷积 (x_j - x_i, x_i) → MaxPool
- 最佳：**83.92%** +- 4.39% (stride=1, K=20, +Depression extra data)
- 变体：DANN 对抗、SupCon 对比、Band SE、空间先验、MixUp、Focal、TTA
- 特点：动态图学习功能连接，MaxPool > Attention

### 03_GAT — 图注意力网络
- 模型：k-NN → 学习注意力权重 softmax → 加权求和
- 最佳：81.79% +- 4.93% (+Depression extra data)
- 特点：Soft attention 被噪声干扰，不如 EdgeConv MaxPool

### 04_MoGE — 混合图专家
- 模型：每通道路由器 → 3 个并行 EdgeConv 专家 → 门控融合
- 最佳：81.05% (=EdgeConv v1 baseline)
- 特点：通道专业化无显著增益，架构复杂度过拟合风险

### 05_RawWaveform — 端到端原始波形
- 模型：TemporalCNN 学习频带 → EdgeConv 空间 → FC
- 最佳：67.85%
- 结论：手工 DE 特征是不可超越的强基线（小样本下）

### 06_LaBraM — 预训练 ViT
- 模型：BEiT-v2 风格 ViT，在大规模 EEG 数据上预训练
- 最佳：83.89% +- 5.75% (stride=3, 200-dim input)
- 注意：需要预训练权重 `labram-base.pth`，放入 `checkpoints/` 目录
- 弱点：高方差 +-5.75%，预训练模型被异质数据（抑郁症）反噬

## 数据格式

### DE 特征 (.npy)
```
_X.npy: (N_windows, 30 channels, 5 bands)  # δ/θ/α/β/γ
_Y.npy: (N_windows,)  # 0=neutral, 1=positive
```

### 原始波形 (.npy)
```
_X.npy: (N_windows, 30 channels, 250 timepoints)  # 250Hz × 1s
_Y.npy: (N_windows,)
```

## 核心发现

1. **DE 特征不可超越**：手工 Butterworth 5 频带滤波是强归纳偏置
2. **EdgeConv > Chebyshev**：动态图碾压固定图 (+1.73%)
3. **MaxPool > Attention**：硬选择在噪声 EEG 中最优
4. **简洁架构最优**：所有正则化技巧（Focal/LabelSmooth/SupCon/DANN）均负向
5. **预训练模型被异质数据反噬**：LaBraM + Depression = −2.45%
6. **从零训练的灵活架构吃异质数据增益**：EdgeConv + Depression = +1.16%

## 运行环境

- Python 3.10+, PyTorch 2.x, CUDA 12.x
- LaBraM 额外需要：timm, einops, h5py
- 数据路径：`--data_dir` 指向包含 `*_X.npy` / `*_Y.npy` 的目录
