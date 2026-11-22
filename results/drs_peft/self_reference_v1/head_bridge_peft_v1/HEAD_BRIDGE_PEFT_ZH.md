# Head/Bridge-aware PEFT

## 协议

本轮从每折 COMMON_WARMUP 起点重新训练，不复用 Full FT 的 Head 或 Bridge 更新。输入固定为 MATR 三折、12 通道、source fit/tune/screen、seed 42、AdamW、8 epoch 和学习率候选 1e-4、1e-3。

Head 参数和 fusion12 参数由脚本动态统计。所有候选都不加入物理特征，也不扩大已有 backbone rank。

## 实际参数与 source screen

| 方法 | Head | fusion12 | Backbone adapter | 总可训练参数 | 三折 screen MAE | 最优 epoch |
|---|---:|---:|---:|---:|---:|---:|
| HEAD_ONLY_TRAIN | 6,337 | 0 | 0 | 6,337 | 0.031318338 | 4 |
| BRIDGE12_ONLY_TRAIN | 0 | 53,248 | 0 | 53,248 | 0.037090115 | 8 |
| HEAD_BRIDGE_TRAIN | 6,337 | 53,248 | 0 | 59,585 | 0.029093287 | 8 |
| HEAD_LORA | 6,337 | 0 | 30,336 | 36,673 | 0.030507117 | 4 |
| HEAD_BRIDGE_LORA | 6,337 | 53,248 | 4,224 | 63,809 | **0.029044688** | 8 |
| BACKBONE_ONLY | 0 | 0 | 36,864 | 36,864 | 0.037442340 | 8 |

所有方法的最佳学习率为 1e-3。Fold 0、1、2 的方向一致。

## 关键比较

- HEAD_ONLY_TRAIN 相对 BACKBONE_ONLY 改善约 0.006124 MAE。
- HEAD_BRIDGE_TRAIN 相对 HEAD_ONLY_TRAIN 进一步改善约 0.002225 MAE。
- HEAD_BRIDGE_LORA 相对 HEAD_BRIDGE_TRAIN 再改善约 0.000049 MAE。
- BRIDGE12_ONLY_TRAIN 只有约 0.000352 MAE 的改善。
- HEAD_BRIDGE_TRAIN 和 HEAD_BRIDGE_LORA 都优于之前 Full FT source screen 参照 0.033634840。

逐电芯比较中，相对 BACKBONE_ONLY：

- HEAD_ONLY_TRAIN 有 41/46 颗电芯改善。
- HEAD_LORA 有 42/46 颗电芯改善。
- HEAD_BRIDGE_TRAIN 有 38/46 颗电芯改善。
- HEAD_BRIDGE_LORA 有 38/46 颗电芯改善。
- BRIDGE12_ONLY_TRAIN 有 45/46 颗电芯改善，但平均改善较小。

## 判断

1. 输出端 Head 是当前适配收益的主要来源。只训练 6,337 个 Head 参数已经明显超过只训练 backbone adapter 的基线。
2. fusion12 单独训练的收益有限。它与 Head 联合训练时产生明显协同。
3. Head + fusion12 在约 60k 参数预算下已经显著优于 36k backbone-only 结构。
4. 在约 36k 预算下，HEAD_LORA 使用 36,673 个参数并明显优于 BACKBONE_ONLY，说明预算应优先分配给 Head，再分配给 late-layer backbone。
5. HEAD_BRIDGE_LORA 的 63,809 参数相对 HEAD_BRIDGE_TRAIN 的额外收益很小。当前没有证据表明必须再加入 backbone adapter 才能获得主要收益。

## 限制

这些结果仍然是 source-only screen，且当前 MATR 队列已经参与过多轮开发。Full FT 参照的训练路径和 Head-aware PEFT 的学习率选择不同，Full FT 只作为历史高容量参照。正式跨协议或 target 泛化尚未执行。

## 产物

- PEFT_SCORES.csv 保存折级指标、模块参数量、显存、训练和推理时间。
- PEFT_TRACE.csv 保存 fit/tune 轨迹。
- PEFT_CELL.csv 保存逐电芯误差。
- PEFT_BUDGETS.csv 保存实际参数预算和 rank。
- PEFT_SUMMARY.csv 保存三折汇总。
- PEFT_COMPLETE.json 保存标签隔离与文件 hash。
