# 单侧方向少样本迁移实验复核

## 当前执行范围

状态为COMPLETE。已评分135个请求槽，待完成0个，失败或不可估计0个。实际适配优化更新0次。

输入合同为VIQ+earlySOH / no-policy。native_multivariate使用每圈12个统计描述量。源模型从policy维度0初始化，SOH锚点及前缀单位来源单列于输入审计。

## 完整未来与远期K曲线

| stage | domain | method | K | window | mae | gain_mae | negative_transfer_ratio | physical_query_cells | protocol_groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3 | MATR | AUTO | 1 | 1001_plus | 0.115883 | -0.00210263 | 0.471429 | 14 | 9 |
| D3 | MATR | AUTO | 1 | all | 0.0294727 | -0.000338426 | 0.566667 | 60 | 39 |
| D3 | MATR | AUTO | 3 | 1001_plus | 0.119948 | -0.00616698 | 0.514286 | 14 | 9 |
| D3 | MATR | AUTO | 3 | all | 0.0311574 | -0.00202315 | 0.5 | 60 | 39 |
| D3 | MATR | AUTO | 5 | 1001_plus | 0.109604 | 0.0041766 | 0.3 | 14 | 9 |
| D3 | MATR | AUTO | 5 | all | 0.030054 | -0.00091969 | 0.583333 | 60 | 39 |
| D3 | MATR | CROSS | 1 | 1001_plus | 0.115062 | -0.00128098 | 0.514286 | 14 | 9 |
| D3 | MATR | CROSS | 1 | all | 0.0295021 | -0.00036786 | 0.493333 | 60 | 39 |
| D3 | MATR | CROSS | 3 | 1001_plus | 0.12328 | -0.00949957 | 0.542857 | 14 | 9 |
| D3 | MATR | CROSS | 3 | all | 0.0317695 | -0.00263522 | 0.516667 | 60 | 39 |
| D3 | MATR | CROSS | 5 | 1001_plus | 0.116268 | -0.00248717 | 0.471429 | 14 | 9 |
| D3 | MATR | CROSS | 5 | all | 0.0304336 | -0.00129931 | 0.55 | 60 | 39 |
| D3 | MATR | RANDOM | 1 | 1001_plus | 0.116095 | -0.00231486 | 0.514286 | 14 | 9 |
| D3 | MATR | RANDOM | 1 | all | 0.0296178 | -0.000483532 | 0.58 | 60 | 39 |
| D3 | MATR | RANDOM | 3 | 1001_plus | 0.110016 | 0.003765 | 0.371429 | 14 | 9 |
| D3 | MATR | RANDOM | 3 | all | 0.0294094 | -0.000275119 | 0.52 | 60 | 39 |
| D3 | MATR | RANDOM | 5 | 1001_plus | 0.108685 | 0.00509574 | 0.214286 | 14 | 9 |
| D3 | MATR | RANDOM | 5 | all | 0.0293649 | -0.000230663 | 0.586667 | 60 | 39 |
| D3 | MATR | ZERO | 0 | 1001_plus | 0.113781 | 0 | 0 | 14 | 9 |
| D3 | MATR | ZERO | 0 | all | 0.0291343 | 0 | 0 | 60 | 39 |

MATR主任务由原54颗目标池中的4个共同可行精确工况组成，固定8颗Query。每个工况获得K颗Support，聚合信息预算为4K。HUST固定39颗Query，Support预算为跨工况池中的K颗。Stanford仅按供应循环坐标列为附加协议。

## 同Query方法差异

| stage | domain | K | window | reference_method | candidate_method | reference_mae | candidate_mae | candidate_gain_over_reference | physical_query_cells | paired_cell_seed_draw_rows | candidate_improved_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3 | MATR | 1 | 1001_plus | AUTO | CROSS | 0.115883 | 0.115062 | 0.000821652 | 14 | 70 | 0.542857 |
| D3 | MATR | 1 | 1001_plus | RANDOM | CROSS | 0.116095 | 0.115062 | 0.00103388 | 14 | 70 | 0.514286 |
| D3 | MATR | 1 | all | AUTO | CROSS | 0.0294727 | 0.0295021 | -2.94335e-05 | 60 | 300 | 0.57 |
| D3 | MATR | 1 | all | RANDOM | CROSS | 0.0296178 | 0.0295021 | 0.000115672 | 60 | 300 | 0.593333 |
| D3 | MATR | 3 | 1001_plus | AUTO | CROSS | 0.119948 | 0.12328 | -0.00333258 | 14 | 70 | 0.514286 |
| D3 | MATR | 3 | 1001_plus | RANDOM | CROSS | 0.110016 | 0.12328 | -0.0132646 | 14 | 70 | 0.3 |
| D3 | MATR | 3 | all | AUTO | CROSS | 0.0311574 | 0.0317695 | -0.000612075 | 60 | 300 | 0.53 |
| D3 | MATR | 3 | all | RANDOM | CROSS | 0.0294094 | 0.0317695 | -0.0023601 | 60 | 300 | 0.526667 |
| D3 | MATR | 5 | 1001_plus | AUTO | CROSS | 0.109604 | 0.116268 | -0.00666377 | 14 | 70 | 0.271429 |
| D3 | MATR | 5 | 1001_plus | RANDOM | CROSS | 0.108685 | 0.116268 | -0.00758291 | 14 | 70 | 0.214286 |
| D3 | MATR | 5 | all | AUTO | CROSS | 0.030054 | 0.0304336 | -0.000379616 | 60 | 300 | 0.576667 |
| D3 | MATR | 5 | all | RANDOM | CROSS | 0.0293649 | 0.0304336 | -0.00106864 | 60 | 300 | 0.536667 |

candidate_gain_over_reference为逐cell、同seed及draw的参考方法MAE减去候选方法MAE，正值支持候选改善。每行明确当前已评分的配对数量。D3仅使用seed42，三训练种子证据由D2和D4分别提供。

## 抽样与训练种子波动

| stage | domain | method | K | window | seed_std_after_averaging_draws | mean_within_seed_task_draw_std | seeds | draw_variance_includes_explicit_repeated_support_slots | independent_units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3 | MATR | AUTO | 1 | all | NA | 0.00043495 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D3 | MATR | AUTO | 3 | all | NA | 0.00289311 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D3 | MATR | AUTO | 5 | all | NA | 0.00115164 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D3 | MATR | CROSS | 1 | all | NA | 0.000490568 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D3 | MATR | CROSS | 3 | all | NA | 0.00415029 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D3 | MATR | CROSS | 5 | all | NA | 0.00154188 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D3 | MATR | RANDOM | 1 | all | NA | 0.000566073 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D3 | MATR | RANDOM | 3 | all | NA | 0.000711951 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D3 | MATR | RANDOM | 5 | all | NA | 0.000499267 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D3 | MATR | ZERO | 0 | all | NA | NA | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |

draw波动在同task与seed内计算，seed波动先对draw平均后计算。重复Support集合与相同adaptation_id保留，独立样本数按物理电芯与工况计数。

## 适配成本

按唯一adaptation_id计，非线性适配器拟合0次，局部正规化求解123次。ADAPTATION_COSTS.csv保留每个唯一状态的实际优化步数、cache_solver、logtime、字典字节与线性化差距，未测量字段保留NA。

当前共有126条唯一状态回执。源参考模型训练次数在REFERENCE_MODELS.csv单列。

所选范围的累计engine计数包含inner与评价，实际非线性优化更新0次。已完成源参考模型3组、训练阶段拟合6次，累计源优化更新5760次。不同阶段汇总共享的参考训练只能计一次。

## 失败与不可估计

当前无已评分结果。

远期窗口没有观测时保留NA。Support下降、Query恶化作为负迁移保留，源开发负结果继续进入已锁定的目标评价流程。

## 优势与风险

优势为共同f0、固定Query、嵌套物理Support、fold-local构基及真实非线性前向评分。四种几何臂与通用LoRA基线分别登记维度、冻结存储和优化成本。

风险包括源115颗与三折的历史暴露、训练集重叠、有限远期支持、局部线性近似误差，以及Support协议覆盖不足。source空间方向不赋予未验证的电化学含义。

现有数字的解释以实际覆盖为限。汇总程序只读取已评分表，未读取目标未来标签数组，也未新增bootstrap或模型运行。数值汇总核验置信度取决于请求槽覆盖与算术一致性。

## 拓展思考

结合K曲线、最强通用基线、CAL2、负迁移与成本判断下一步，单独的梯度捕获比例不作为方法成功依据。

## 你可能想知道

输入或超参数选择只使用源内部inner任务。目标一经评分登记为已暴露，当前结果不能重新作为未见目标测试。
