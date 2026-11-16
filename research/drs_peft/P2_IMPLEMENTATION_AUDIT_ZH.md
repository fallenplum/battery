# P2 首个真实搜索单元独立实现审计

## 审计范围

审计对象为LoRA、rank4、seed42、trial0，adapter LR为0.0001，共用bridge/head LR为0.0003。该单元属于P2源内LR搜索，源训练85颗、source-meta-validation 30颗。当前记录用于核验已完成单元的算子、参数、谱系、选权与重放。P2完整方法矩阵和目标域少样本结果仍待后续执行。

独立审计通过CPU执行，GPU在进程环境中隐藏，本次审计优化器更新为0。真实搜索单元训练60个epoch，每epoch11个batch，合计660次优化更新。

精简证据见[独立审计JSON](../../results/drs_peft/P2/P2_INDEPENDENT_IMPLEMENTATION_AUDIT.json)。运行产物位于runtime/drs_peft/P2/search/lora/rank4/trial0/seed42，公开小型结果位于results/drs_peft/P2/search/lora/rank4/trial0/seed42。

## 目标与选择规则

loss按每颗物理电池的全部已观测未来点计算MAE，再在batch内对电池等权平均。各电池的有效观测数量参与监督mask归一化。训练时的query子集由该batch的源监督支持确定，预测入口始终使用预先固定的101–5000公共网格，共4900个query。

30颗元验证电池共有21827个有效未来观测，保存预测包独立重算的cell-mean MAE为**0.025850552718330356**。ID顺序与冻结元验证角色逐项一致。

60个epoch日志连续，updates从11递增至660，loss与梯度范数均有限。选权规则为本单元source-meta-validation cell-mean MAE的首次严格最小值。所选epoch为**55**，对应605次训练更新。完整搜索单元消耗660次更新。最终TRAIN_STATE的所有活跃AdamW state.step均为660，磁盘CHECKPOINT_SELECTED与TRAIN_STATE中的selected模型tensor逐项相等。

## 参数与共同起点

| 组件 | 实际可训练参数 |
| --- | --- |
| Bridge | 56,832 |
| 连续cycle-query Head | 6,337 |
| LoRA Q/V adapter | 36,864 |
| 合计 | **100,033** |

共有12个Q/V矩阵，每个原矩阵为384×384，rank为4，单矩阵adapter参数为3072。alpha为4，实际缩放alpha/r为1。W0的冻结参数指纹在训练与独立重建中一致。

CPU重建使用已固定的官方Moirai权重和对应seed的正式P0 B0所选bridge/head。加载后完整模型指纹与P1_BASES所记录的源checkpoint逐字节一致。源scaler的parsed JSON与P0完全一致。文件SHA差异来自Linux LF与Windows CRLF换行。

| 身份 | SHA256 |
| --- | --- |
| 官方Moirai权重 | cc46831272ea07e99d78fe05c09e730c7c76a1b272cddbdaf5173dae371cb238 |
| P0 B0所选checkpoint文件 | a368f07bd3aaf7b43c409b5309133e2ed0f6b90dec4e3564d01707da01a2704d |
| 共同B0完整模型tensor指纹 | 63f04b3f5ad3d3a09e6501a19f580837bb23624fd7dd1eea0899e9233fe711f6 |
| 冻结参数tensor指纹 | a30f5c06e0d594189d9fcec7e13e3b302b7fec004d5e69da4ca42deb5d123488 |
| P2所选checkpoint文件 | d556efe2601654c728dd9997289fd52791171688b6d80c15a804a65526b204b5 |
| P1基与梯度均值文件 | 1b7c8acdc9b8f4696cee2f2f6dd28674bab2280e578f3180f26886195a70f15f |

已核验16个私有文件记录、12个公开文件副本及13项源码/config快照。逐文件SHA在JSON中保留，source commit为a0f651f。

## 磁盘重建与CPU重放

训练单元通过独立读取磁盘checkpoint、重新构造模型、严格加载selected state后的GPU重放，记录的最大绝对误差为0。

本次另在CPU重建官方主干、P0共同初态及已训练LoRA，读取已允许的K100前缀与source-meta-validation预测包，在完整公共query网格重放30颗电池。CPU/GPU最大绝对差为**1.4066696166992188e-05**，满足已有P0独立CPU重放容差atol=1e-4、rtol=0。

CPU重算cell-mean MAE为0.025850589400353296，保存GPU结果为0.025850552718330356。

## LoRA-GA源梯度均值权重

独立读取12层各85颗源电池的缓存梯度，并逐文件核验SHA。所有12层mean_gradients逐bit等于逐物理电池的平均梯度，所有12层group_mean_gradients逐bit等于47个exact protocol等权后的组均值。

- mean_gradients对85颗物理电池等权，每颗电池内部的loss先按已观测未来点归一。该缓存提供LoRA-GA初始化所需的完整source objective梯度。
- group_mean_gradients先在每个exact protocol内部对电池等权，再对47个protocol等权。该缓存用于P1的组等权mean-SVD描述。
- 两种均值的相对Frobenius差在12层中为**7.06097274%–12.42562220%**，权重定义在数值上有可观测差异。

每层一致性与梯度数组SHA已保留在JSON。后续LoRA-GA单元仍需完成实际训练与自身重放闭环。

## 信息预算与证据边界

RUN_CONFIG和真实receipt的final target Query标签使用计数均为0。独立审计只解压完整cohort缓存的早期前缀和metadata成员，未来标签来自已允许的source-meta-validation预测包。源码中源监督、选权与评分均通过冻结的source/meta角色索引。

优势为共同起点、真实训练参数、660次更新、磁盘state、冻结tensor、源梯度均值及独立CPU重放形成一致的证据链。风险包括source-meta-validation同时参与epoch与LR选择，阶段指标具有选择条件性。真实P1的label-shuffle捕获未弱化，退化结局特异性仍缺少支持。剩余方法、三seed正式比较及目标域少样本结果需要各自完成对应阶段。

算子与该单元谱系核验的置信度为高。本审计的结论范围为实现一致性。
