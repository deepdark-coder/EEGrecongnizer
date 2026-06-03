# conbine

这个目录只放“组合调用/融合”相关脚本，不修改原有 `code/` 与 `EEG-Conformer/` 源码。

## 设计原则

- 不回改原模型训练脚本。
- 需要适配的地方，尽量在新脚本中通过包装函数、适配器类或局部重写来完成。
- 统一先导出预测文件，再做融合，避免不同模型直接耦合。

## 统一预测文件格式

所有模型最终都建议导出为 `.csv`，字段如下：

```text
subject_id,label,prob_0,prob_1,pred_class,source
```

如果能做到更细粒度对齐，也支持额外加入：

```text
chunk_id,start_idx,end_idx
```

其中：

- `subject_id`：被试 ID，例如 `HC1040`
- `label`：真实标签，`0/1`
- `prob_0` / `prob_1`：两类概率
- `pred_class`：预测类别
- `source`：模型名，例如 `edgeconv` / `labram` / `eeg_conformer`

## 推荐融合方式

### 1. Soft Voting

最稳妥，直接对各模型概率做加权平均：

```text
P = w1 * P1 + w2 * P2 + w3 * P3
```

适合：

- 三个模型都已经单独训练好
- 希望先低风险拿一点提升
- 各模型输出已经能对齐到同一批 `subject_id`

### 2. Orthogonal Stacking

不是简单平均，而是先把后续模型对前面模型“重复解释”的部分剥掉，再做二层线性融合。

直观上：

- 先保留主模型信息
- 再让其它模型只补充“剩余增量”
- 降低多个相似模型一起投票时的冗余

这个仓库里尤其适合：

- `EdgeConv` 和 `LaBraM` 都基于 DE 特征窗口
- `EEG-Conformer` 走原始波形，信息更独立

所以一个经验顺序可以是：

1. `EdgeConv`
2. `LaBraM`
3. `EEG-Conformer`

## 当前脚本

- `ensemble.py`：读取多个预测 `.csv`，做 soft voting 与 orthogonal stacking
- `export_edgeconv_predictions.py`：从 EdgeConv 结果目录导出统一预测文件
- `export_labram_predictions.py`：直接调用 LaBraM 模型做预测导出，不改原始源码
- `export_eeg_conformer_predictions.py`：直接调用 EEG-Conformer 模型做预测导出，不改原始源码

## 建议流程

1. 先分别导出三路预测文件
2. 检查 `subject_id` 是否一致
3. 先跑 `soft voting`
4. 再跑 `orthogonal stacking`
5. 比较验证集或交叉验证结果，再决定是否固定权重
