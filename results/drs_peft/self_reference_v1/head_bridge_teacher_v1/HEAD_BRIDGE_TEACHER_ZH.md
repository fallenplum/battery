# Head/Bridge-aware Full FT 教师功能消融

本实验从同折 warmup 和 Full FT epoch 8 快照出发，仅进行 source screen 推理。输入固定为 12 通道，数据固定为 MATR 三折。Head、Bridge 和 backbone 更新均来自同折 source fit 产生的 Full FT 教师更新。

| 方法 | 三折 screen MAE | 说明 |
|---|---:|---|
| FULL_FT_12 | 0.033634840 | 完整 Full FT 更新 |
| HEAD_ALL_BACKBONE | 0.033716146 | Head 加全部 TSFM backbone 更新，排除 fusion12 |
| HEAD_BRIDGE | 0.033769781 | Head 加 fusion12 |
| HEAD_LATE_BACKBONE | 0.033808769 | Head 加 encoder 后三层 |
| HEAD_ONLY | 0.033829761 | 只保留 Head |
| BRIDGE_ONLY | 0.037497011 | 只保留 fusion12 |
| WARMUP_ONLY_12 | 0.037537646 | 无适配 |

Head-only 已恢复 warmup 到 Full FT 收益的大部分。Bridge-only 接近 warmup，单独输入融合更新的作用较小。Head+Bridge 比 Head-only 改善约 0.000059979。Head+全部 backbone 比完整 Full FT 差约 0.000081305，说明 Head、Bridge 和 backbone 之间仍有协同。

本实验是教师更新诊断。下一轮输出端 PEFT 必须从 warmup 重新训练 Head、Bridge 和低秩 backbone 参数，不能直接复用本轮 Full FT 的更新。
