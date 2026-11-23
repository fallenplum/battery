# MICH/RWTH 新确认性候选矩阵审计

本轮只读取供应 pkl 的元数据和 1–100 周期输入字段。未读取 100 周期后的容量值，未进行目标评分。

| Family | 原始文件 | 前100完整 | 资格电芯 | 目标口径 |
|---|---:|---:|---:|---|
| MICH | 40 | 40 | 40 | supplied cycle observed SOH |
| RWTH | 48 | 48 | 48 | supplied cycle observed SOH |

主比较固定为 HEAD_ONLY_TRAIN、HEAD_LORA、HEAD_BRIDGE_LORA。WARMUP_ONLY、LORA_12、FULL_FT_TUNED 只作为未来标签公开后的回顾性补充矩阵，不参与确认性方法选择。

MICH 与 RWTH 仍需分别构建 prefix-only 12 通道 cache、固定输入窗口、写入 prediction lock 后，才可进入新 family 预测。
