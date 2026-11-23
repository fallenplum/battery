# CALCE 与 SNL 新 family 输入、标签与资格口径审计

## 共同设置

| 项目 | 当前口径 |
|---|---|
| 输入前缀 | 供应 pkl 的周期 1–100 |
| 模型输入 | 每周期 12 个 NativeBridge 统计通道 |
| 电压网格 | 2.8–3.3 V，64 点，仅用于构造 Q(V) 后统计 |
| 放电段 | 负电流有效段，当前 blind cache 使用 `current_tolerance=0.0` 的 active-segment 规则 |
| 参考形变 | 同一电芯前 1–10 个有效周期的逐点中位 Q(V) |
| 漂移通道 | 当前周期 Q(V) 减去自身前 1–10 周期参考曲线，再按 nominal Ah 归一化 |
| Anchor | 周期 100 放电容量最大值除以 nominal Ah |
| 预测网格 | 101–5000，缺失周期保持 observed=false |
| 未来标签 | 周期大于 100 的 `max(discharge_capacity_in_Ah) / nominal_capacity_in_Ah` |

## CALCE

- 纳入 12 颗电芯，CS2 与 CX2 各 6 颗。
- CS2 nominal capacity 为 1.1 Ah，CX2 nominal capacity 为 1.35 Ah。
- 每颗前缀 100 个周期均通过当前 2.8–3.3 V 曲线资格。
- 输入 cache receipt 记录 `prefix_future_labels_read=0`、`target_future_labels_read=0`。
- 未来容量标签只在 `PREDICTIONS_LOCKED.json` 写入后由独立评分步骤读取。

## SNL

- 纳入 28 颗电芯，包含 LFP、NCA、NMC。
- nominal capacity 范围为 1.1–3.2 Ah。
- 每颗前缀 100 个周期均通过当前 2.8–3.3 V 曲线资格。
- 当前纳入样本中 LFP 只有 1 颗，LFP 分组误差不具备稳定估计能力。
- 输入 cache receipt 记录 `prefix_future_labels_read=0`、`target_future_labels_read=0`。
- 未来容量标签只在 `PREDICTIONS_LOCKED.json` 写入后由独立评分步骤读取。

## 与 MATR 的可比性边界

CALCE 与 SNL 使用供应 pkl 重新构造 12 通道输入。MATR 的主实验使用已冻结的原生 Q(V) cache 和其数据审计管线。两者在电压窗口、放电段选择、参考曲线重建和原始坐标认证上存在差异，因此本轮结果用于跨 family 结构筛选，不能视为完全同构物理坐标下的精确对比。

## 资格判断

这两个 family 没有进入当前 Head-aware 方法选择、source fold、outer 或 target 记录。BatteryLife 通用基线流程可能接触过它们，因此本轮属于对当前 Head-aware 方法选择未暴露的新 family blind 评价，不能声称项目级完全未知。

