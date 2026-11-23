# E0 MICH/RWTH 七方法全矩阵

## 口径

七个方法均从 MATR source fit/tune 选择，预测文件先锁定，随后读取 MICH/RWTH 供应 pkl 的 100 周期后容量标签。

主方法包含 `HEAD_ONLY_TRAIN`、`HEAD_LORA`、`HEAD_BRIDGE_TRAIN`、`HEAD_BRIDGE_LORA`。补充方法包含 `WARMUP_ONLY`、`LORA_12`、`FULL_FT_TUNED`。

## 总体 MAE

| Family | WARMUP | LoRA-12 | HEAD_ONLY | HEAD_LORA | HEAD_BRIDGE_TRAIN | HEAD_BRIDGE_LORA | FULL_FT |
|---|---:|---:|---:|---:|---:|---:|---:|
| MICH | 0.189165 | 0.189106 | 0.178455 | 0.178555 | 0.175075 | **0.174892** | 0.178896 |
| RWTH | 0.158055 | 0.157887 | 0.097575 | 0.094010 | 0.068699 | **0.067604** | 0.098911 |

## 资格偏差

- MICH 原登记窗口为 3.0–4.2V。active discharge 段的共同最大支持约 4.179V，本轮收缩到 3.0–4.15V。
- RWTH 原登记窗口为 3.5–3.9V。active discharge 段共同支持不足，本轮收缩到 3.5–3.83V，并允许每周期至少 2 个电压点插值，缺失周期保留 mask。40/48 颗进入 cache，8 颗因有效 Q(V) 支持不足排除。
- 因此 MICH/RWTH 结果用于根因审计和候选筛选，不能解释为与 MATR 完全同构的物理坐标确认性结果。

## 资源记录

逐 fold、逐方法的训练时间、显存、参数量和预测时间见各 family 的 `E0_LEDGER.csv`。逐电芯和四窗口结果见 `E0_CELL.csv` 与 `E0_WINDOW.csv`。

