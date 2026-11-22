# Full FT 教师更新功能消融

## 协议

本实验执行 Stage 1 功能消融。所有方法从同折 `FULL_FT_EPOCH0.pt` warmup 状态出发，使用同折 epoch 8 Full FT 更新的指定子集进行 source screen 推理。输入固定为 12 通道，数据固定为 MATR 三折和既定 source fit、tune、screen 角色。

本实验不训练新参数，因此 `trainable_parameters=0`，`training_seconds=0`。`teacher_update_parameters=13,999,369` 只表示来源 Full FT 的参数规模。每个方法的显存峰值、推理时间、逐电芯 MAE/MSE/RMSE、checkpoint hash 和标签读取计数均写入 CSV 或 JSON 收据。

这一步属于教师更新诊断。它用于判断哪些更新组对 source screen 预测有功能贡献，不能直接作为部署 PEFT 结果。

## 三折 screen 结果

表中 MAE、RMSE、MSE 是先按电芯计算，再对电芯求平均。`screen_pooled_*` 按全部有效观测点汇总，作为补充口径。

| 方法 | MAE | RMSE | MSE | 相对 Full FT 的 MAE 差 |
|---|---:|---:|---:|---:|
| FULL_FT_12 | **0.033634840** | 0.056685 | 0.003577 | 0 |
| NO_QK | 0.033635796 | 0.056690 | 0.003577 | +0.000000955 |
| NO_LAST_LAYER | 0.033646019 | 0.056704 | 0.003579 | +0.000011179 |
| NO_ATTN | 0.033650891 | 0.056712 | 0.003580 | +0.000016051 |
| NO_VO | 0.033651664 | 0.056711 | 0.003580 | +0.000016824 |
| NO_FFN | 0.033685695 | 0.056770 | 0.003587 | +0.000050855 |
| NO_BRIDGE | 0.033716145 | 0.056824 | 0.003593 | +0.000081305 |
| HEAD_BRIDGE | 0.033769780 | 0.056909 | 0.003604 | +0.000134940 |
| NO_HEAD | 0.037489411 | 0.061870 | 0.004198 | +0.003854571 |
| FFN_ONLY | 0.037514164 | 0.061900 | 0.004201 | +0.003879324 |
| FFN_ATTN | 0.037534730 | 0.061921 | 0.004203 | +0.003899890 |
| ATTN_ONLY | 0.037542844 | 0.061932 | 0.004205 | +0.003908004 |
| LATE3 | 0.037578244 | 0.061969 | 0.004209 | +0.003943404 |
| EARLY3 | 0.037632870 | 0.062027 | 0.004216 | +0.003998030 |
| WARMUP_ONLY_12 | 0.037537645 | 0.061922 | 0.004204 | +0.003902805 |

## 逐折方向

- `NO_FFN` 三折均比 Full FT 变差，MAE 增量分别为 0.000078151、0.000042863、0.000031551。
- `NO_ATTN` 三折均变差，增量为 0.000028442、0.000012598、0.000007113。
- `NO_QK` 的总体差异极小。Fold 0 略有改善，Fold 1 和 Fold 2 变差，不能登记为稳定收益。
- `NO_HEAD` 三折均明显变差，44/46 个 screen 电芯的误差高于 Full FT。
- `HEAD_BRIDGE` 保留了大部分 Full FT 收益，但仍比完整更新差 0.000134940。

## 判断

1. Head 更新是当前 Full FT source screen 收益的必要组成。去掉 Head 后性能接近 warmup-only，单独保留 FFN、Attention 或 FFN+Attention 也无法恢复收益。
2. Bridge 更新提供次于 Head 的可测贡献。去掉 Bridge 后 MAE 增加约 0.000081305。
3. FFN 和 Attention 的更新对完整模型有小幅协同贡献，但单独保留它们不能形成有效预测修正。`NO_FFN`、`NO_ATTN`、`NO_VO` 和 `NO_LAST_LAYER` 的损失都远小于去掉 Head 的损失。
4. Q/K 更新的独立贡献在本轮很小。`NO_QK` 的三折方向不一致，不能据此删除 Q/K，也不能把它登记为有效简化。
5. 参数更新能量排名与功能重要性排序不同。FFN 的更新能量最高，但 Head 的功能消融影响最大。

## 下一阶段约束

Stage 2 应把 Head 和 Bridge 作为结构化 PEFT 的必要组成，再在约 36,864 个可训练参数预算内比较 FFN、Attention、后三区域和非均匀 rank 分配。所有候选必须重新训练低维参数，不能直接复用本轮 Full FT 更新。30 通道特征、电池特征调制和 target/outer 评价继续保持关闭。

## 产物

- `FUNCTIONAL_ABLATION_SCORES.csv` 保存折级指标、时间、显存、参数量和标签计数。
- `FUNCTIONAL_ABLATION_CELL.csv` 保存每颗 screen 电芯的误差。
- `FUNCTIONAL_ABLATION_SUMMARY.csv` 保存三折汇总。
- `FUNCTIONAL_ABLATION_COMPLETE.json` 保存协议、checkpoint hash 和文件 hash。
