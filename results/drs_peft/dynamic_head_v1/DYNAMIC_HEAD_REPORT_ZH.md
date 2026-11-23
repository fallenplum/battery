# 动态 Head 校准 source-only 结果

## 目的

在共同 seed42 warmup 起点、12 通道前缀和相同 source fit/tune/screen 划分下，检验电芯自身前 100 圈统计是否能选择少量输出 Head 更新方向。

## 结果

| 方法 | 参数量 | Fold 0 MAE | Fold 1 MAE | Fold 2 MAE | Screen MAE |
|---|---:|---:|---:|---:|---:|
| `HEAD_ONLY_TRAIN` | 6337 | 0.033611 | 0.029539 | 0.030806 | **0.031318** |
| `CONST_HEAD_R4` | 228 | 0.033846 | 0.029644 | **0.030664** | **0.031385** |
| `DYNAMIC_HEAD_R4` | 228 | 0.035263 | 0.030205 | 0.030857 | 0.032109 |

`DYNAMIC_HEAD_R4` 在三个 fold 都高于 `CONST_HEAD_R4`，平均 MAE 高出 0.000724。它也高于 `HEAD_ONLY_TRAIN`，平均 MAE 高出 0.000790。真实前缀条件在这套低秩最终 Head 参数化中没有带来 source screen 收益。

## 信息与边界

- `DYNAMIC_HEAD_R4` 和 `CONST_HEAD_R4` 都训练 228 个参数，学习率候选为 `1e-4`、`1e-3`，检查点为 2、4、8 epoch。
- `HEAD_ONLY_TRAIN` 从同一 warmup 状态重新训练，使用相同 fit/tune/screen 角色和学习率候选。
- 条件向量只来自 12 通道前 100 圈输入的掩码均值和最后有效值。
- 三个方法的 `target_future_labels_read=0`、`outer_labels_read=0`。
- 该轮只做 source screen。按预设停止条件，不进入 source outer 或 target 回顾性评分。

## 判断

当前证据不支持继续扩大这个动态输出条件器。`CONST_HEAD_R4` 已接近 6337 参数的静态 Head，而真实状态条件反而稳定变差。这个结果不能证明电池状态永远没有输出端信息，只能说明当前 24 维前缀摘要、rank-4 最终层更新和训练预算下没有形成可迁移的 source screen 优势。

