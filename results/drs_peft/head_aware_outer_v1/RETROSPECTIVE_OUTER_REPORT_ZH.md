# Head-aware retrospective source outer evaluation

## 协议

使用 e123_folds_v1 的三折 outer_test_ids。每折模型仍从当前 self_reference_v1 source fit/tune 重新选择 checkpoint，输入固定为 12 通道，seed 42。所有 outer predictions 在读取 outer future labels 前锁定。当前队列属于历史暴露 retrospective evaluation。

## 结果

| 方法 | 三折 outer MAE |
|---|---:|
| HEAD_BRIDGE_LORA | **0.031417986** |
| HEAD_BRIDGE_TRAIN | 0.031486774 |
| FULL_FT_TUNED | 0.033899394 |
| HEAD_LORA | 0.032960764 |
| HEAD_ONLY_TRAIN | 0.034063545 |
| LORA_12 | 0.039628496 |
| WARMUP_ONLY | 0.039716936 |

Head-aware 方法在 outer source 电芯上保持优于 LORA-12 和 WARMUP_ONLY。HEAD_BRIDGE_LORA 比公平 FULL_FT_TUNED 低约 0.002481408 MAE。

## 边界

- outer labels 只在 OUTER_PREDICTIONS_LOCKED.json 生成后读取。
- target future labels 未读取。
- 当前结果不是 blind confirmation，不能替代新的电芯或新的工况队列。
- 仍然只有 seed 42，不能登记多随机种子稳定性。
