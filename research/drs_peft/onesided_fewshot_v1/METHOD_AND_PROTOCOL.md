# 单侧方向的真实电池少样本校准

本轮执行用户提供的 `CODEX_BATTERY_NEXT_PLAN_AFTER_ABC.md`，父提交为 `e3d0602634e2365b0a9a375ffefec0229dbefb06`。本文件登记执行中的具体选择，结果由实际产物确定。

## 输入与共同起点

主比较名称为 **VIQ+earlySOH / no-policy**。模型从初始化采用零维 policy 接口，Bridge/Head 共 59585 个参数。绝对电流、容量和前缀相对容量响应使用声明的额定容量归一化。额定容量用于物理单位转换。其数值没有额外拼接到模型输入。

native multivariate Bridge 每圈消费 12 维描述量。前 100 圈的 64 点 VIQ 和合法 earlySOH anchor 是可用信息。预测网格固定为原始圈号 101 至 5000。未来长度、未来观测掩码、EOL 与协议标识仅用于评分或分组。

每个 fold 和 seed 从官方预训练 Moirai-small 重训 no-policy warmup，然后拟合 rank-4 静态核心。warmup 与 static 分别最多 120 epoch，各自仅依本 fold inner MAE 选取最早严格最优模型。旧 policy checkpoint 仅可用于工程检查。

最终目标域参考模型使用全部 115 颗源电池。每个 seed 的 warmup 和 static epoch 分别取三个源 CV 折的最佳 epoch 中位数。按此固定预算训练，使用最后 epoch。目标 Query 不参与选择。

## 几何方向与支持集适配

在最终静态参考函数上重新采集 fit 电池的逐电池梯度。固定秩空间的 CORE、OUT、IN 和 NORMAL 四块经全模型 Frobenius 正交分解，各使用最多 16 个跨层共享系数。OUT 保留输入侧 V 并扩展输出方向。若有效维数不足，报告实际维数并增加共同最小维度对照。

支持集每颗电池最多使用 512 个确定性等间隔真实观测点，损失按物理电池等权。几何与 CAL2 使用 FP64 Cholesky Ridge，实际正则系数为登记倍率乘以 Gram 平均对角元，并以 1e-12 为下限。更新阻尼只在源 inner 上选择。本轮不追加重线性化。

标准 LoRA 使用 rank 1。LoRA-FA 使用 rank 2，固定随机行正交 A 并训练 B。两者各训练 9216 个参数，保留共同静态源核心，从零函数增量开始。两种低秩基线执行真实 Support AdamW 适配。几何 16 参数与低秩 9216 参数的竞争分别报告参数量。

## 选择与评价

D2 使用三个源工况 CV 折及 seed 42、123、2026。每折固定 outer Query 20 颗，inner Query 8 颗。K 为 0、1、3、5，非零 K 有 5 个预登记嵌套 draw。每个 fold、seed、method、K 的超参数只使用本 fold inner Query。重复支持集缓存复用，仍保留全部登记 draw 的评分。

D3 作为小谱对照，仅运行 seed 42 的三个折。候选为相同 OUT 16 维空间，AUTO、CROSS 与 RANDOM 使用匹配的最多 8 个方向。CROSS 使用同一已知协议内部所有有序异电池对，协议等权，仅保留正特征值。不可估计时登记 NA，后续 D4 继续执行。

D4 在源阶段结束后锁定 OUT、CAL2、ZERO 和 source-inner 选出的最佳通用低秩方法。目标超参数由全部 source-inner 结果汇总选择。MATR 四个精确协议任务共 8 颗固定 Query，每任务使用 K 颗 Support，因此聚合结果耗费 4K 颗 Support。HUST 使用 39 颗固定 Query，从 38 颗独立 Support 中选 K 颗。目标预测先保存，随后评分并登记 Query 已暴露。源负收益或目标负迁移均保留完整登记矩阵。

## 方法身份与风险

低秩切空间分块、固定随机 A、梯度子空间及小核适配均有文献先例。本轮证据范围为方向定位和真实少样本有效性。

共同参考函数、固定 Query 和独立 Support 有利于归因。主要风险来自历史源数据已参与研究、有限支持集抽样波动、局部线性化误差、远期删失及原始曲线压缩。统计按物理 Query 电池、协议、draw 和 seed 分开报告，新增 bootstrap 次数为零。

若所有登记方法在源 outer 的非零 K 平均增益均不为正，且 MATR 与 HUST 各自均无跨 K 平均正收益方法，再触发用户允许的有限接口探针。接口选择只使用源 fit/inner，已评分目标 Query 不参与再次选择。

## 文献依据

- 固定秩切空间及法向分解依据 Gao 与 Absil 的方法定义。[原论文](https://link.springer.com/article/10.1007/s10589-021-00328-w)
- A 与 B 的输入输出作用以及固定 A 的既有证据来自 Asymmetry in Low-Rank Adapters。[原论文](https://proceedings.mlr.press/v235/zhu24c.html)
- 固定 A、训练 B 的基线来自 LoRA-FA。本轮采用合同规定的随机行正交 A。[原论文](https://arxiv.org/html/2308.03303)
- 源更新子空间用于少样本适配已有 SubGD 研究。[原论文](https://proceedings.mlr.press/v199/gauch22a.html)
- 梯度导出固定低秩基及小核心已有 LoRA-SB 研究。[原论文](https://arxiv.org/html/2411.19557)
