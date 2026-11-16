# P2 源内受控比较

本阶段为FULL_REGISTERED_PRIMARY_AND_PARAMETER_CURVES，正式单元66个。P2门槛状态为**NEGATIVE**。所有方法从对应seed的正式P0 B0共用bridge/head与官方W0启动。每方法接受seed42的三LR、每次60epoch搜索，选定LR后所有三个seed各训练120epoch。搜索与正式更新分别记录。

同rank主比较为r4，实际参数匹配core rank为56。实际参数曲线状态为COMPLETE，额外rank曲线状态为COMPLETE。P3/P4大矩阵放行状态为False。W-SVD、LoRA与LoRA-GA对照保留在配对比较中，XS含原方法Sigma和Gaussian核心初始化。

主指标为每physical cell完整已观测未来MAE后对cell平均。配对区间以cell及exact protocol分别bootstrap，source seed方向单列。source-meta-validation同时用于LR选择与本阶段报告，区间以本次选择为条件。最终目标Query未来标签访问为0。

优势为共同B0起点、相同优化更新与LR试验次数、真实参数曲线和完整阴性记录。风险为源数据规模、选择条件化的不确定性及外域子空间失配。当前结果的置信度须结合三seed方向、区间和外域冻结后结果判断。
