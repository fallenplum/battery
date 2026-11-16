# P1 源退化梯度诊断

本次为seed2026的真实源数据诊断，源训练85颗、47个exact protocol，元验证30颗、15组。全部基只由源训练构造，优化器更新为0。最终目标Query标签未参与梯度、选模或评分计算。

主基由每组等权、组内每cell等权的梯度能量左右协方差构建，另有mean-SVD、组内去均值、W0-SVD、随机正交及源标签打乱控制。完整单cell梯度在gitignored runtime缓存，校验和与身份在P1_CACHE_MANIFEST.json，小型基在P1_BASES.pt。

主rank为4。单seed捕获条件满足状态为True，bootstrap基重合均值为0.9280794364147842。H1当前标PARTIAL，正式门槛仍需三source seed汇总。单cell梯度秩受token预算约束，当前谱形与投影不能单独认证电池特异退化机制。

每个收集role的首尾cell在缓存写完后独立重新前后向，逐层比较梯度并保存P1_GRADIENT_REPLAY.json。此次复算覆盖120个层梯度，不将有限复算描述为全量重放。

source checkpoint、head与bridge的before/after指纹相同，临时requires_grad与train/eval标志均已恢复。source名义容量取值为[1.100000023841858]。绝对单位控制是在同一checkpoint上的输入单位扰动，容量恒定时缺少跨容量可比性证据。

各层和各group的投影、主角度、逐cell梯度支持及bootstrap在对应CSV中保留。打乱控制若同样捕获方向，应作为个体退化特异性不足的证据。优势为真实梯度、独立源工况留出和多控制，风险为有限组内重复、单seed、固定head与桥接诱发的共同方向。工程身份保持的置信度高，H1的科学支持仍需完整阶段结果。
