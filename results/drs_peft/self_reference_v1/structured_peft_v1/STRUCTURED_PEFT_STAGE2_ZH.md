# Stage 2 参数预算匹配结构化 PEFT

## 锁定协议

本轮固定同一 Moirai-small backbone、同一折 COMMON_FOLD*.pt warmup、MATR 三折、12 通道、source fit/tune/screen、MAE 损失、AdamW、训练 8 epoch、学习率候选 1e-4 和 1e-3。所有候选只在 backbone 的已登记线性层加入 LoRA，Bridge 和 Head 维持共同 warmup 状态。

目标预算为 36,864 个可训练参数。除 FFN-only 因矩阵宽度为 1,408、无法整除目标预算外，其余候选达到 36,864。FFN-only 使用 36,608 个参数，差异为 256 个参数，低于 0.7%。

## source screen 结果

| 方法 | 参数量 | 三折 MAE | 三折 RMSE | 三折 MSE | 最优 epoch | 最优学习率 |
|---|---:|---:|---:|---:|---:|---:|
| LATE_LAYER_GROUP | 36,864 | **0.037441956** | 0.061822 | 0.004191 | 8 | 0.001 |
| FFN_ONLY | 36,608 | 0.037442550 | 0.061823 | 0.004191 | 8 | 0.001 |
| FFN_ATTN_ALL | 36,864 | 0.037443882 | 0.061824 | 0.004192 | 8 | 0.001 |
| GROUPWISE_RANK | 36,864 | 0.037444961 | 0.061825 | 0.004192 | 8 | 0.001 |
| ATTN_ONLY | 36,864 | 0.037446028 | 0.061826 | 0.004192 | 8 | 0.001 |
| LORA_12 | 36,864 | 0.037451947 | 0.061832 | 0.004193 | 8 | 0.001 |

相对 LORA-12 的 MAE 差异如下。

- LATE_LAYER_GROUP：-0.000009991
- FFN_ONLY：-0.000009397
- FFN_ATTN_ALL：-0.000008065
- GROUPWISE_RANK：-0.000006987
- ATTN_ONLY：-0.000005919

所有结构化候选的三折方向都优于 LORA-12，但平均差异约为 1e-5，尚未达到明显超过的证据强度。每个候选的最佳 epoch 都是 8，最佳学习率都是 1e-3。

逐电芯配对结果显示，LATE_LAYER_GROUP、FFN_ONLY 和 FFN_ATTN_ALL 各有 45/46 颗 screen 电芯优于 LORA-12。改善幅度很小，因此不能据此登记结构化 PEFT 已经突破。

## 研究判断

1. 在当前 36k 预算下，更新位置和非均匀 rank 分配存在一致但很小的 source screen 信号。
2. LATE_LAYER_GROUP 是本轮最优候选，但其优势不足以排除训练波动或已有队列暴露影响。
3. Stage 2 的结构完全无效判断不成立。Stage 2 的结构性突破已建立判断也不成立。
4. 按合同进入一次有限预算阶梯，只对 LATE_LAYER_GROUP 测试约 36k、140k 和 500k。输入特征、loss、优化器、学习率候选和训练轮数保持不变。

## 产物

- STRUCTURED_PEFT_SCORES.csv 保存每折 screen 指标、最佳 epoch、训练和推理时间、显存、参数量与 hash。
- STRUCTURED_PEFT_TRACE.csv 保存 fit/tune 选择轨迹。
- STRUCTURED_PEFT_CELL.csv 保存逐电芯 screen 误差。
- STRUCTURED_PEFT_BUDGETS.csv 保存各方法实际 rank 和参数预算。
- STRUCTURED_PEFT_SUMMARY.csv 保存三折汇总。
- STRUCTURED_PEFT_COMPLETE.json 保存标签隔离和文件 hash 收据。
