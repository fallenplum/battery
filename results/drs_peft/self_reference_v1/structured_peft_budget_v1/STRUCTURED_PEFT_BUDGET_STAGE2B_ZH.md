# Stage 2b 预算阶梯实验

## 协议

本轮只延续 Stage 2 中 source tune 表现最好的 LATE_LAYER_GROUP。输入、MATR 三折、12 通道、共同 warmup、MAE、AdamW、学习率候选 1e-4 和 1e-3、8 个 epoch 均保持不变。只改变适配器预算。

## 结果

| 预算 | 实际参数量 | 三折 MAE | 三折 RMSE | 三折 MSE |
|---|---:|---:|---:|---:|
| 约 36k | 36,864 | 0.037441956 | 0.061822 | 0.004191 |
| 约 140k | 131,328 | 0.037434325 | 0.061814 | 0.004190 |
| 约 500k | 503,424 | **0.037432240** | 0.061812 | 0.004190 |

三种预算的最佳 epoch 都是 8，最佳学习率都是 1e-3。Fold 0、1、2 的改善方向一致，但从 36k 增加到 500k 只带来约 0.000009715 的 MAE 改善。Full FT 的 source screen MAE 为 0.033634840，500k 结构化 PEFT 仍相差约 0.003797400。

## 判断

1. 36k 到 500k 的容量增加没有带来接近 Full FT 的预测能力。
2. 当前候选的主要限制不宜归因于单纯的参数预算不足。
3. 仅扩大同一 LATE_LAYER_GROUP 的 rank 不足以形成方法突破。
4. 后续应进入 Stage 3 的 Full FT 教师子空间学习，使用 source fit 产生的固定 SVD 基底，在新的训练过程中只学习低维系数。直接截断 Full FT 的结果只能作为教师诊断参照。

## 产物

- STRUCTURED_PEFT_BUDGET_SCORES.csv 保存每折指标、预算、最佳 epoch、显存和 hash。
- STRUCTURED_PEFT_BUDGET_TRACE.csv 保存 fit/tune 轨迹。
- STRUCTURED_PEFT_BUDGET_CELL.csv 保存逐电芯误差。
- STRUCTURED_PEFT_BUDGETS.csv 保存实际预算。
- STRUCTURED_PEFT_BUDGET_SUMMARY.csv 保存汇总。
- STRUCTURED_PEFT_BUDGET_COMPLETE.json 保存协议和文件 hash 收据。
