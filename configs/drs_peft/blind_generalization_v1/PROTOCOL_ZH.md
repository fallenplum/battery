# Head-aware 新 family blind 泛化协议

## 固定候选

本轮只比较三个已经登记的方法：

- `HEAD_ONLY_TRAIN`
- `HEAD_LORA`
- `HEAD_BRIDGE_LORA`

停止动态 Head、固定 SVD basis 和额外 backbone rank。

## 新队列

首轮使用 BatteryLife 中未参与当前 Head-aware 方法选择的 CALCE family。只纳入前 100 个供应周期完整、2.8–3.3V 放电窗口支持充分且存在 100 周期后轨迹的电芯。CALCE 的通用 BatteryLife 基线接触风险单独记录，因此本轮结论称为“对当前 Head-aware 方法选择未暴露的新 family blind 评价”。

## 信息边界

- source fit/tune 未来标签只用于已登记的 MATR source 选择。
- CALCE 前缀构建只读取前 100 周期。
- 先写 `BLIND_LOCK.json`，再生成目标预测。
- 预测文件和 `PREDICTIONS_LOCKED.json` 写完后，才读取 CALCE 的 100 周期后容量轨迹。
- CALCE 未来标签不参与方法、学习率或 checkpoint 选择。

## 评价

报告每个方法的逐电芯未来轨迹 MAE、RMSE、分时距窗口误差、有效目标点数和方法间配对差异。保留 chemistry、temperature、protocol 字段或从供应文件名解析的分组信息，不按目标标签挑选分组。

