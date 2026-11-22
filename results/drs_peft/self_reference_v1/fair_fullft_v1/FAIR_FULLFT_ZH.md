# 公平 Full FT source 对照

## 协议

Full FT 与 Head-aware PEFT 使用同一 COMMON_WARMUP、同一 MATR 三折、同一 source fit/tune/screen、12 通道、seed 42、batch size 8、AdamW、梯度裁剪 1.0、8 epoch 和学习率候选 1e-4、1e-3。Full FT 按 tune MAE 在 epoch 2、4、8 中选择，Head-aware PEFT 使用相同选择规则。

## 结果

| 方法 | 参数量 | 三折 screen MAE |
|---|---:|---:|
| WARMUP_ONLY | 0 | 0.037537646 |
| BACKBONE_ONLY | 36,864 | 0.037442340 |
| FULL_FT_TUNED | 13,999,369 | 0.031195065 |
| HEAD_ONLY_TRAIN | 6,337 | 0.031318338 |
| HEAD_LORA | 36,673 | 0.030507117 |
| HEAD_BRIDGE_TRAIN | 59,585 | 0.029093287 |
| HEAD_BRIDGE_LORA | 63,809 | **0.029044688** |

公平 Full FT 选择的 epoch 为 Fold 0 第 8 个 epoch，Fold 1 和 Fold 2 第 4 个 epoch，三折最佳学习率均为 1e-3。

## 判断

1. 公平 Full FT 从此前 0.033635 的固定参照改善到 0.031195，说明原先 Full FT 与 Head-aware PEFT 的训练选择确实不完全可比。
2. 公平后 Head_ONLY_TRAIN 仍略优于 FULL_FT_TUNED，HEAD_LORA、HEAD_BRIDGE_TRAIN 和 HEAD_BRIDGE_LORA 的 source screen MAE 进一步下降。
3. HEAD_BRIDGE_LORA 相对 FULL_FT_TUNED 的 source screen MAE 差为约 -0.002150。
4. 这些结果仍是 source-only 队列内部结果，不能替代 outer 或 target 验证。

## 收据

FAIR_FT_SCORES.csv 保存折级指标、显存、训练时间、checkpoint hash 和标签读取计数。FAIR_FT_TRACE.csv 保存 tune 选择轨迹，未保存模型权重。
