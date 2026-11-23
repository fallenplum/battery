# Head-aware 新 family blind 泛化结果

## 评价边界

本轮事先固定三个候选：`HEAD_ONLY_TRAIN`、`HEAD_LORA`、`HEAD_BRIDGE_LORA`。source 选择只使用 MATR 的已登记 fit/tune 角色。新 family 的前 100 周期输入先冻结，所有 fold 和方法预测文件写入并锁定后，才读取新 family 的 100 周期后容量轨迹。

CALCE 和 SNL 均未出现在当前 Head-aware 方法选择、source fold、outer 或 target 记录中。BatteryLife 通用基线流程可能接触过这些 family，因此这里的结论限定为**对当前 Head-aware 方法选择未暴露的新 family blind 评价**，不宣称项目级完全未知。

## 总体结果

| Family | 电芯数 | HEAD_ONLY_TRAIN | HEAD_LORA | HEAD_BRIDGE_LORA |
|---|---:|---:|---:|---:|
| CALCE | 12 | 0.146751 | 0.148454 | **0.142376** |
| SNL | 28 | 0.069270 | 0.070317 | **0.065604** |

两个新 family 的总体结果都由 `HEAD_BRIDGE_LORA` 最低。相对它，`HEAD_ONLY_TRAIN` 在 CALCE 和 SNL 上分别高 0.004375 和 0.003666，`HEAD_LORA` 分别高 0.006078 和 0.004714。

## 分组结果

CALCE 的 CS2/CX2 两组均由 `HEAD_BRIDGE_LORA` 最低。

| CALCE 组 | HEAD_ONLY_TRAIN | HEAD_LORA | HEAD_BRIDGE_LORA |
|---|---:|---:|---:|
| CS2 | 0.159620 | 0.161578 | **0.159289** |
| CX2 | 0.133882 | 0.135331 | **0.125462** |

SNL 的结果存在化学体系差异。LFP 只有一颗通过当前严格前缀处理的电芯，NCA 和 NMC 覆盖更多电芯。

| SNL 化学体系 | HEAD_ONLY_TRAIN | HEAD_LORA | HEAD_BRIDGE_LORA |
|---|---:|---:|---:|
| LFP | 0.088432 | **0.082743** | 0.094665 |
| NCA | **0.086054** | 0.088446 | 0.087781 |
| NMC | 0.043113 | 0.042819 | **0.030704** |

因此不能把 `HEAD_BRIDGE_LORA` 概括为所有化学体系上的稳定赢家。它在两个 family 的总体均值上领先，但 SNL 分组方向不一致。

## 数据与锁定

- CALCE 纳入 12 颗电芯，排除 1 颗前 100 周期不完整的 CX2 电芯。
- SNL 纳入 28 颗电芯。其余供应文件在当前共同 2.8–3.3V、放电有效段和 12 通道前缀资格规则下未进入本轮队列。
- 两个 family 都使用供应 pkl 重建前 100 周期的 12 通道 NativeBridge 统计输入。该输入处理与 MATR 原生冻结 Q(V) 管线不完全同构。
- `BLIND_SCORE_RECEIPT.json` 均记录预测锁定哈希和 `target_future_labels_read=1`。标签读取发生在 `PREDICTIONS_LOCKED.json` 之后。

## 判断

当前证据支持保留 `HEAD_BRIDGE_LORA` 作为下一轮主要候选，但不足以删除 `HEAD_ONLY_TRAIN` 或 `HEAD_LORA`。跨新 family 的总体方向一致，化学体系分组仍显示结构依赖。下一轮如果继续，应扩大真正未参与通用基线训练的留出来源，或增加独立 family，而不是按本轮目标结果重新挑选结构。

