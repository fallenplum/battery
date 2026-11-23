# Dynamic Head calibration protocol

## Question

在共同 warmup 起点和 12 通道输入下，前 100 圈 VIQ 统计能否选择少量输出 Head 更新方向，使源域未参与拟合的 screen 电芯优于静态 Head-aware 参照。

## Scope

- 仅使用已登记的 12 通道输入和 source fit/tune/screen 划分。
- 不加入物理特征，不扩大 TSFM backbone rank，不读取 outer 或 target 未来标签。
- 每个 seed、每个 fold 从对应 `COMMON_FOLD*.pt` 独立恢复。
- tune 只用于学习率和 checkpoint 选择，screen 只在选择后评分。

## Methods

- `WARMUP_ONLY`：共同 warmup，冻结 backbone、Bridge、Head。
- `HEAD_ONLY_TRAIN`：现有静态 Head 全量训练参照。
- `DYNAMIC_HEAD_R4`：冻结 warmup Head，在最终 Head 输出层加入低秩动态更新。更新系数由该电芯前 100 圈 12 通道统计生成。
- `CONST_HEAD_R4`：同样的动态更新参数预算，系数由固定全一探针生成，不读取电芯状态。

动态更新只作用于 Head 的最后线性映射，保持 anchor-preserving 的参考值相减结构。共享方向、状态映射和常数偏置均从 source fit 学习，目标域不做测试时更新。

## Primary criteria

首要指标是 source screen 的等电芯 MAE。只有 `DYNAMIC_HEAD_R4` 同时优于 `CONST_HEAD_R4` 和 `HEAD_ONLY_TRAIN`，并且提升不依赖单一 fold，才进入 source outer 回顾性复核。若真实状态与常数对照无差异或更差，则停止该候选，不继续扩大动态网络。

## Required records

记录每个方法的实际可训练参数量、学习率、最佳 checkpoint、训练损失、tune MAE、screen MAE、每电芯 MAE、预测修正 RMS、状态映射输入哈希以及未来标签读取计数。只保留汇总表、锁定回执和必要脚本，权重留在私有 runtime。

