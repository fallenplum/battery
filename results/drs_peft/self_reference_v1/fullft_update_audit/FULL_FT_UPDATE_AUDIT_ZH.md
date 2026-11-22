# Full FT 更新结构审计

## 目的

本审计从每个源域折的锁定 `COMMON_FOLD*.pt` 出发，使用 12 通道输入，在 source fit 电芯上进行 8 个 epoch 的 Full FT，并在 epoch 0、2、4、8 保存模型快照。每个快照与同折 epoch 0 的 warmup 状态相减，得到真实监督更新 `ΔW_FT`。

本轮只使用 MATR 的 source fit、tune、screen 角色。审计收据记录 `target_future_labels_read=0` 与 `outer_labels_read=0`。

## 运行设置

- Backbone、Bridge、Head 起点来自同折锁定的 `COMMON_FOLD0/1/2.pt`。
- Full FT 显式重新打开全部参数的梯度，包含 TSFM backbone、Bridge 和 Head。
- 优化器为 AdamW，学习率 `1e-4`，权重衰减 0，梯度裁剪范数 1，训练 8 epoch。
- 可训练参数数量为 `13,999,369`。
- 更新几何保存奇异值、`rank99`、stable rank、Frobenius 范数和相对 warmup 范数。

## 预测参照

同一锁定 warmup 起点的 source screen 全轨迹 MAE 如下。数值为三折未加权均值。

| 方法 | Fold 0 | Fold 1 | Fold 2 | 三折均值 |
|---|---:|---:|---:|---:|
| Warmup-only-12 | 0.041338474 | 0.036551768 | 0.034722695 | 0.037537645 |
| LoRA-12，同一 warmup | 0.041175540 | 0.036489053 | 0.034691224 | 0.037451939 |
| Full FT-12，epoch 8 | 0.036716446 | 0.032236867 | 0.031951207 | **0.033634840** |

在这组 source screen 中，Full FT 相对 warmup-only 改善 `0.003902805`，相对同起点 LoRA-12 改善 `0.003817099`。这只是源域内部参照，不能当作目标域确认结果或理论上限。

## 更新能量

epoch 8 的三折参数更新按组汇总如下。组内 Frobenius 范数先平方求和再开方，能量占比按平方范数计算。

| 参数组 | 跨折汇总范数 | 能量占比 | 平均 rank99 | 平均 stable rank |
|---|---:|---:|---:|---:|
| FFN fc2 | 1.336857 | 22.88% | 191.06 | 2.79 |
| FFN fc1 | 1.286223 | 21.18% | 162.94 | 2.21 |
| FFN fc_gate | 1.283468 | 21.09% | 178.11 | 2.42 |
| Attention out_proj | 0.797925 | 8.15% | 61.06 | 1.85 |
| Attention v_proj | 0.740383 | 7.02% | 57.67 | 1.78 |
| Attention q_proj | 0.739990 | 7.01% | 96.17 | 2.26 |
| Attention k_proj | 0.738373 | 6.98% | 126.72 | 3.61 |
| Head | 0.578750 | 4.29% | 3.56 | 1.23 |
| 12 通道 Bridge fusion | 0.321620 | 1.32% | 15.67 | 1.45 |
| 其他 | 0.079789 | 0.08% | 0.39 | 0.39 |
| 30 通道 Bridge fusion | 0 | 0 | 0 | 0 |

FFN 三组合计约 **65.1%** 的更新能量。Attention 投影合计约 **29.2%**。在编码器层中，索引 5 的层总更新范数最高，为 1.203824，其次是索引 4 和索引 3，分别为 1.143794 与 1.139116。这里的层汇总只统计 TSFM encoder 层，不包含 Head 和 Bridge。

## 跨折稳定性

公开收据不包含完整更新向量，因此没有把参数方向余弦误称为已测量结果。当前稳定性文件比较的是每个参数的 Frobenius 范数 profile。各主要参数组的 profile cosine 为 0.9923 到 1.000，绝对夹角多数低于 5°，最大约 7.1°。这说明不同 source fold 的更新**幅度分配模式**较稳定，但不能推出更新方向本身稳定。

## 研究判断

1. Full FT 在当前 source 任务上产生了明确的 backbone 更新，q、k、v、out 和 FFN 都不是零更新。因此，后续研究可以把真实 Full FT 更新作为 PEFT 候选结构的监督参照。
2. 更新能量主要集中在 FFN，再分布到 attention 投影。只围绕 Q/V 的低秩适配不能直接覆盖 Full FT 的主要能量来源。
3. `rank99` 在大多数矩阵组明显高于极小 rank，说明“普通 rank-1 或 rank-4 必然足够”没有得到这轮审计支持。但 stable rank 仍较低，更新谱可能存在长尾集中，是否能被小子空间有效逼近必须通过重构误差与预测性能共同检验。
4. 目前只证明了源域 Full FT 更新的几何形状和 source screen 参照，尚未证明某个 SVD、梯度或电池特征子空间能够在独立协议上替代 Full FT。

## 产物与一致性

- `UPDATE_GEOMETRY.csv` 保存每个参数、折和里程碑的更新几何。
- `UPDATE_GROUP_SUMMARY.csv` 保存 epoch 8 的参数组汇总。
- `UPDATE_LAYER_GEOMETRY.csv` 与 `UPDATE_LAYER_SUMMARY.csv` 保存层级汇总。
- `UPDATE_STABILITY.csv` 保存跨折范数 profile 稳定性。
- `CHECKPOINT_HASHES.csv` 保存 12 个 source checkpoint 的 SHA-256。
- `COMPLETE.json` 与 `STABILITY_COMPLETE.json` 保存标签隔离和文件哈希收据。

运行中曾存在一份旧的嵌套私有快照目录，其 backbone 更新为零。核对后确认它来自修正前的审计路径，已移除且未用于当前公开结果。当前公开 CSV 与顶层 corrected 快照对应，q_proj 三折更新范数分别为 0.417892、0.438489、0.425062。
