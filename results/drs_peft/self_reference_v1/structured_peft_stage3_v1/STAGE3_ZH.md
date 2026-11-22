# Stage 3 Full FT 教师子空间诊断

## 协议

本轮使用同一 MATR 三折、12 通道、共同 warmup、source fit/tune/screen、AdamW、8 epoch 和学习率候选 1e-4、1e-3。对每个 fold 的 42 个 q/k/v/out/FFN 二维线性层构造 rank 30 的左右基底。

Fixed SVD Basis 的基底来自同折 source fit 产生的 Full FT epoch 8 更新。Random Basis 使用同样维度的确定性随机正交基底。两种方法都从 warmup 起点重新训练低维核心 C。每层核心为 30×30，实际可训练参数为 37,800。

该实验是 teacher-informed 诊断。Fixed SVD Basis 使用了 Full FT 产生的基底，因此它不等同于部署时可直接训练的 PEFT 方法。

## source screen 结果

| 方法 | 参数量 | 三折 MAE | 三折 RMSE | 三折 MSE |
|---|---:|---:|---:|---:|
| Fixed SVD Basis | 37,800 | **0.037442369** | 0.061822 | 0.004191 |
| Random Basis | 37,800 | 0.037464906 | 0.061845 | 0.004194 |
| Standard LoRA-12 参照 | 36,864 | 0.037451947 | 0.061832 | 0.004193 |
| Full FT exact 参照 | 13,999,369 | 0.033634840 | 0.057085 | 0.003366 |

Fixed SVD Basis 相对 LoRA-12 的 MAE 改善约 0.000009579。Random Basis 相对 Fixed SVD Basis 变差约 0.000022537。逐电芯比较中，Random Basis 有 1 颗电芯略优于 Fixed SVD Basis，45 颗更差。

## 判断

1. Fixed SVD Basis 的 source screen 结果略优于 LoRA-12，Random Basis 在几乎全部电芯上更差，说明 Full FT 教师方向包含弱的可利用信息。
2. Fixed SVD Basis 的结果仍接近 36k 结构化 PEFT，距离 Full FT 约 0.0038 MAE。教师方向没有在当前低维核心训练中恢复 Full FT 的主要收益。
3. 该结果不足以支持电池特征调制或部署级动态子空间。下一步需要控制 teacher leakage，把基底构造和适配训练严格分离，并在未用于 Full FT 基底构造的 source 工况上验证。
4. 公开仓库只保存结果、轨迹、预算和收据。真实基底张量与 Full FT checkpoint 保留在私有运行目录，不进入仓库。

## 产物

- STAGE3_SCORES.csv 保存折级指标、时间、显存、最佳 epoch 和 hash。
- STAGE3_TRACE.csv 保存 fit/tune 训练轨迹。
- STAGE3_CELL.csv 保存逐电芯 screen 误差。
- STAGE3_BUDGETS.csv 保存基底 rank 与可训练参数量。
- STAGE3_SUMMARY.csv 保存三折汇总。
- STAGE3_COMPLETE.json 保存标签隔离与文件 hash。
