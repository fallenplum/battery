# M4 完整 Methodology 的 seed42 正式结果

六个固定划分各运行150 epochs，正式更新12450次。完整fold0复验额外2100次，单epoch工程smoke另14次，实际优化更新合计14564次。fresh-batch重复与数学oracle均为0次优化更新。验证集全轨迹 cell-mean MAE 首次最小值选择权重，选权冻结后评分测试集。

## 完整轨迹与时距

|范围|指标|单位|cell mean|protocol macro|支持电芯|协议|观测数|
|---|---|---|---|---|---|---|---|
|main_oof|horizon_1001_1000000_mae|SOH|0.0829638039|0.0856235031|40|19|10614|
|main_oof|horizon_1001_1000000_rmse|SOH|0.0859831765|0.0881833496|40|19|10614|
|main_oof|horizon_101_200_mae|SOH|0.00537210603|0.00792659671|169|77|16865|
|main_oof|horizon_101_200_rmse|SOH|0.00581530902|0.00867831007|169|77|16865|
|main_oof|horizon_201_500_mae|SOH|0.0101244148|0.0146156331|168|76|49795|
|main_oof|horizon_201_500_rmse|SOH|0.0130499844|0.0190123046|168|76|49795|
|main_oof|horizon_501_1000_mae|SOH|0.0450655648|0.0446560865|151|61|46157|
|main_oof|horizon_501_1000_rmse|SOH|0.0552985275|0.0512080189|151|61|46157|
|main_oof|r200_500_absolute_error|SOH/cycle|6.45485993e-05|8.48677461e-05|151|61|302|
|main_oof|r500_1000_absolute_error|SOH/cycle|0.000146507053|0.000152684831|42|19|84|
|main_oof|trajectory_mae|SOH|0.0232710748|0.0215282057|169|77|123431|
|main_oof|trajectory_rmse|SOH|0.0385274196|0.032855653|169|77|123431|
|unseen_protocol_sensitivity|horizon_1001_1000000_mae|SOH|0.20930777|0.183884133|14|6|6267|
|unseen_protocol_sensitivity|horizon_1001_1000000_rmse|SOH|0.230511652|0.200041006|14|6|6267|
|unseen_protocol_sensitivity|horizon_101_200_mae|SOH|0.00598265507|0.0112138224|46|15|4571|
|unseen_protocol_sensitivity|horizon_101_200_rmse|SOH|0.00648281942|0.0122214159|46|15|4571|
|unseen_protocol_sensitivity|horizon_201_500_mae|SOH|0.0140976463|0.0240971982|45|14|13378|
|unseen_protocol_sensitivity|horizon_201_500_rmse|SOH|0.0177151128|0.0306939558|45|14|13378|
|unseen_protocol_sensitivity|horizon_501_1000_mae|SOH|0.0504334467|0.0594644259|40|11|14096|
|unseen_protocol_sensitivity|horizon_501_1000_rmse|SOH|0.0608001824|0.0669442593|40|11|14096|
|unseen_protocol_sensitivity|r200_500_absolute_error|SOH/cycle|7.47844584e-05|0.000111525072|40|11|80|
|unseen_protocol_sensitivity|r500_1000_absolute_error|SOH/cycle|0.000167432375|0.000178408484|14|6|28|
|unseen_protocol_sensitivity|trajectory_mae|SOH|0.0482477825|0.0472276771|46|15|38312|
|unseen_protocol_sensitivity|trajectory_rmse|SOH|0.0747149734|0.0693738856|46|15|38312|

## 六单元与选权

训练目标固定为trajectory MAE + 0.1 × (rate MAE × 1000) + 0.001 × persistence penalty。trajectory项先在每颗电芯的全部有效未来观测上求均值。rate项先在每颗电芯的受支持窗口上求均值，再在有窗口支持的电芯间平均。日志train_loss为该epoch训练batch总目标的均值。

|split|epochs|selected epoch|validation MAE|最终epoch train loss|selected epoch train loss|updates|
|---|---|---|---|---|---|---|
|cell_fold_0|150|124|0.0248964731|0.0152968931|0.0168081585|2100|
|cell_fold_1|150|127|0.0226838865|0.0157187759|0.0215514326|2100|
|cell_fold_2|150|97|0.022485055|0.0140685886|0.0237837124|2100|
|cell_fold_3|150|60|0.0235031913|0.0140471556|0.0280696677|2100|
|cell_fold_4|150|122|0.018542422|0.0168209929|0.0223239928|2100|
|unseen_protocol_sensitivity|150|117|0.0165536238|0.0191227297|0.0253531923|1950|

## 历史同数据对照

|范围|模型|全轨迹MAE|全轨迹RMSE|
|---|---|---|---|
|main_oof|M1|0.0187619638|0.0315785462|
|main_oof|M2|0.0189455988|0.0319231453|
|main_oof|M3|0.0232683617|0.0379553466|
|main_oof|M4_FULL_METHOD|0.0232710748|0.0385274196|
|unseen_protocol_sensitivity|M1|0.0321878464|0.0495096143|
|unseen_protocol_sensitivity|M2|0.027802756|0.0428946103|
|unseen_protocol_sensitivity|M3|0.046896474|0.0716033439|
|unseen_protocol_sensitivity|M4_FULL_METHOD|0.0482477825|0.0747149734|

## EOL80

|范围|模型|Harrell C|可比较对|风险集电芯|观测事件|右删失|
|---|---|---|---|---|---|---|
|main_oof|M4_FULL_METHOD|0.826715565|14179|169|169|0|
|main_oof|M1|0.861132661|14179|169|169|0|
|main_oof|M2|0.859228436|14179|169|169|0|
|main_oof|M3|0.824881868|14179|169|169|0|
|unseen_protocol_sensitivity|M4_FULL_METHOD|0.804263566|1032|46|46|0|
|unseen_protocol_sensitivity|M1|0.838178295|1032|46|46|0|
|unseen_protocol_sensitivity|M2|0.843023256|1032|46|46|0|
|unseen_protocol_sensitivity|M3|0.82751938|1032|46|46|0|

## 查询范围与截断

速率极值覆盖固定100–5000查询轴，单位为SOH/1000 cycles，除以1000得到SOH/cycle。观测监督点按每颗电芯的真实future mask选择。超出随访范围的查询始于该电芯最后有效容量观测之后。all查询还包含100锚点及随访内缺测位置。

表中比例按查询点计数。raw负值表示未截断积分趋势小于0，观测截断表示raw趋势加小幅观测残差后超出[0,1.2]。

|范围|查询区域|电芯|查询点|raw<0比例|观测截断比例|rate最小/1000|rate最大/1000|
|---|---|---|---|---|---|---|---|
|main_oof|all_queries_100_5000|169|828269|0.606691|0.604994|0.00469507|6.22004|
|main_oof|beyond_cell_last_followup|169|704630|0.713144|0.711151|0.063613|6.22004|
|main_oof|observed_future|169|123431|0|0|0.00471795|1.33943|
|unseen_protocol_sensitivity|all_queries_100_5000|46|225446|0.666421|0.665689|0.00289849|4.13869|
|unseen_protocol_sensitivity|beyond_cell_last_followup|46|187079|0.802137|0.801228|0.136467|4.13869|
|unseen_protocol_sensitivity|observed_future|46|38312|0.00467217|0.00480267|0.00291086|1.25721|

100–5000范围的有限、非负速率和单调积分属于数值性质。超过实际随访的速率与趋势缺少同范围的测量校准。


## 机制与限制

保存215个测试角色电芯的机制回执。rho、写入率、真实间隔、漂移方向、两层B和Bprime的L2变化、速率极值及查询一致性见mechanism_summary.csv。

全部169颗电芯已经参与历史研究。本轮结果为固定问题下的内部探索性证据，历史M1/M2/M3对照沿用已保存预测。固定seed42没有测量训练种子方差。配对bootstrap仅条件于这些已拟合预测。

六个正式划分完成后，fold0从seed42重新初始化重复完整150epochs，额外2100次优化更新独立计数。完整训练逐值复现检查为PASS，逐项结果见TRAINING_REPRODUCTION.json。该检查覆盖同环境同seed的一组划分，跨种子方差仍未测量。旧七臂矩阵M2与NR的完整训练逐值重复为0/6，根因仍未认证。

M4首次工程首批检查曾出现预测与loss逐值一致、25组参数梯度逐值失配。失败源码、合同与回执已归档。后续工程改动、重新注册及首批检查结论见FAILURE_HISTORY.json和SMOKE_QA.json。

loaded Q(V)及其导数表示带负载响应。物理分解与写入诊断属于模型内部量，电化学LLI/LAM机制归因仍需独立测量。
