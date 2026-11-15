# M4 完整方法的执行与最终结论

## 执行结论

**M4_FULL_METHOD 已完成六组 seed42 正式训练，每组150轮。主OOF轨迹MAE为0.023271，未见协议MAE为0.048248，误差数值均高于历史M1和M2。** 主OOF与历史M3接近，当前结果未显示新完整方法的预测精度增益。

|模型|主OOF MAE|主OOF RMSE|未见协议 MAE|未见协议 RMSE|
|---|---|---|---|---|
|历史M1|0.018761964|0.031578546|0.032187846|0.049509614|
|历史M2|0.018945599|0.031923145|0.027802756|0.042894610|
|历史M3|0.023268362|0.037955347|0.046896474|0.071603344|
|本次M4|0.023271075|0.038527420|0.048247783|0.074714973|

误差单位为SOH绝对值。主OOF覆盖169颗、77个协议组、123431个实际未来观测，协议敏感性覆盖46颗、15个协议组、38312个观测。每颗电芯先独立计算全轨迹误差，再在电芯间平均。M4协议宏平均MAE分别为0.021528206和0.047227677。

EOL80的Harrell C分别为0.826715565和0.804263566，两组真实事件支持分别169和46颗。四个时距的MAE、RMSE、有效分母、逐折loss、选权及条件配对统计已完整保存。[完整统计报告](../results/unified_matr_methodology_full_seed42/analysis/FULL_MODEL_FINAL_RESULTS_ZH.md)

## 方法合同落实

- I1复用冻结的共同4C放电2.8–3.3V、64点Q(V)，保留20维global、64维residual、8窗各26维local及其掩码。稳健分解和原有全局、局部响应编码保持注册口径。
- I2按当前物理响应、前一有效圈的稳健因果中位状态漂移、局部质量、已知policy及真实cycle gap预测逐圈逐通道rho。第一有效圈直接初始化，缺失位置的内部状态延续。两层history均在卷积后、SSD前真实执行B写入调制，WB零初始化。
- I3输出非负有效损失率，以100圈为锚，在真实整数cycle网格上作固定梯形积分。观测残差幅度为0.03，最终SOH范围为[0,1.2]。头部速率单位为SOH/1000 cycles，除以1000得到SOH/cycle。

合同未指定的实现细节在训练前固定并登记。rho网络为172→64→64，末层初始偏置为logit(0.9)。质量向量32维，写入gate为96→16，WB为64→16且无偏置，两层history均接入。总参数154621。持续性penalty沿用有效转移的逐cycle变化平方均值及0.001系数。[原始完整合同](../research/M4_FULL_METHODOLOGY_RUN_ONLY_ZH.md) [模型与训练登记](../results/unified_matr_methodology_full_seed42/MODEL_CONTRACT.json)

## 训练与核验

|阶段|单元|轮数|优化器更新|
|---|---|---|---|
|工程smoke|第0折|1|14|
|正式评估|五折及协议敏感性共6组|每组150|12450|
|完整训练复验|第0折重新初始化|150|2100|
|合计|||14564|

正式训练使用AdamW、batch8、lr=3e-4和weight_decay=0.01。总loss为trajectory MAE + 0.1 × rate MAE × 1000 + 0.001 × persistence penalty。验证集cell-mean全轨迹MAE的首次最小值选择权重，完整运行150轮后封存选权并评分测试集。

第0折完整重复的20项比较全部逐值一致，覆盖150轮训练与验证记录、最终模型、optimizer、RNG、选中权重及全部预测和机制张量。同环境单折的完整复验提供直接的训练重复证据。[完整复验](../results/unified_matr_methodology_full_seed42/analysis/TRAINING_REPRODUCTION.json)

修复后的独立验收共28475项通过，含15项结构指纹正负控制。六组正式权重及复验权重均通过正常CUDA入口重放，测试预测最大绝对差为0.0。冻结数据、输入和标签张量指纹、fold-local scaler、150轮预算、first-min选权和结果算术均已核验。[最终独立验收](../results/unified_matr_methodology_full_seed42/RUN_VERIFICATION.json)

工程首批曾出现预测和loss一致、25组参数梯度存在微差。仅修动态持续性的scan仍有间歇失配，随后将M4实际使用的各scan封装为同设备autograd节点，前向与初始化保持逐值一致。20次fresh首批及独立12次检查通过后才开始正式训练，失败源码与回执保留。[工程失败与修复记录](../results/unified_matr_methodology_full_seed42/FAILURE_HISTORY.json)

独立验收脚本还修复了容器树指纹缺少边界编码的问题，增加类型、长度及结束标记并重新执行七份选中权重的完整验收。此前数值重放回执及旧脚本公开保留，登记为验收工具缺口的被替代版本。[验收修补记录](../results/unified_matr_methodology_full_seed42/verification_attempts/)

## 远期失效与具体反例

1001圈之后，主OOF的MAE为0.082964，支持40颗、10614点。未见协议的MAE为0.209308，支持14颗、6267点。

**未见协议中的MATR_b1c2出现184/2136个实际观测点的预测下界截断，单电芯全轨迹MAE为0.277551。** 其实际随访到2238圈，观测范围内未截断积分趋势最低为−0.216555。相同电芯的历史M1、M2、M3误差分别约0.091737、0.056273和0.290624，当前结果保留了长寿命电芯外推的具体失败案例。

全部实际观测点中，主OOF截断为0/123431，协议敏感性为184/38312，后者集中在上述一颗。超出各电芯随访至5000圈的查询中，截断分别为501098/704630及149893/187079。100–5000查询的最大速率为6.22004 SOH/1000 cycles，未截断趋势最低约−19.7147。超过随访的预测缺少同范围测量校准。[逐电芯查询诊断](../results/unified_matr_methodology_full_seed42/analysis/query_range_cell_diagnostics.csv) [查询汇总](../results/unified_matr_methodology_full_seed42/analysis/query_range_summary.csv)

## 优势与风险

优势为数据、参数图、真实B写入、训练预算和预测可追溯，且本轮已获得同环境完整训练逐值重复及独立统计重算证据。

风险为主指标误差高于历史M1/M2，协议外推存在严重个体失效。全部169颗具有历史研究暴露，单seed42覆盖的训练随机性有限。物理响应分解和内部写入诊断支持计算图解释，电化学LLI/LAM机制归因仍需要独立测量。

置信度评级为本轮执行与数值复算高，跨种子、跨队列泛化及模块因果贡献低。配对bootstrap仅条件于这些已拟合预测。

## 拓展思考

当前完整方法结果为后续逐一检验I2动态写入和I3积分贡献提供了封存基准。MATR_b1c2可用于明确追踪远期误差来源，后续研究需要持续保留其历史暴露信息。

## 你可能想知道

选中权重、可恢复训练状态、逐电芯轨迹、速率、EOL和B写入诊断均位于新结果目录。正常训练入口为train_methodology_full.py，独立评分与复算入口为两个scripts根目录脚本。[全部新结果](../results/unified_matr_methodology_full_seed42/) [训练入口](../scripts/unified_matr/train_methodology_full.py) [分析入口](../scripts/analyze_methodology_full_seed42.py) [复算入口](../scripts/verify_methodology_full_seed42.py)
