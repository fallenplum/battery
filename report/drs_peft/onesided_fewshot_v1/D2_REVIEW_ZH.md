# 单侧方向少样本迁移实验复核

## 当前执行范围

状态为COMPLETE。已评分318个请求槽，待完成0个，失败或不可估计0个。实际适配优化更新2460次。

输入合同为VIQ+earlySOH / no-policy。native_multivariate使用每圈12个统计描述量。源模型从policy维度0初始化，SOH锚点及前缀单位来源单列于输入审计。

## 完整未来与远期K曲线

| stage | domain | method | K | window | mae | gain_mae | negative_transfer_ratio | physical_query_cells | protocol_groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D2 | MATR | CAL2 | 1 | 1001_plus | 0.135827 | -0.0220466 | 0.571429 | 14 | 9 |
| D2 | MATR | CAL2 | 1 | all | 0.036895 | -0.0077607 | 0.603333 | 60 | 39 |
| D2 | MATR | CAL2 | 3 | 1001_plus | 0.124265 | -0.0104847 | 0.485714 | 14 | 9 |
| D2 | MATR | CAL2 | 3 | all | 0.0344105 | -0.00527621 | 0.636667 | 60 | 39 |
| D2 | MATR | CAL2 | 5 | 1001_plus | 0.119739 | -0.00595834 | 0.3 | 14 | 9 |
| D2 | MATR | CAL2 | 5 | all | 0.0338713 | -0.00473699 | 0.643333 | 60 | 39 |
| D2 | MATR | CORE | 1 | 1001_plus | 0.117283 | -0.00350269 | 0.442857 | 14 | 9 |
| D2 | MATR | CORE | 1 | all | 0.0307674 | -0.00163311 | 0.593333 | 60 | 39 |
| D2 | MATR | CORE | 3 | 1001_plus | 0.111949 | 0.00183161 | 0.428571 | 14 | 9 |
| D2 | MATR | CORE | 3 | all | 0.0318769 | -0.00274259 | 0.553333 | 60 | 39 |
| D2 | MATR | CORE | 5 | 1001_plus | 0.111424 | 0.0023564 | 0.342857 | 14 | 9 |
| D2 | MATR | CORE | 5 | all | 0.033132 | -0.00399772 | 0.61 | 60 | 39 |
| D2 | MATR | IN | 1 | 1001_plus | 0.113446 | 0.000334059 | 0.328571 | 14 | 9 |
| D2 | MATR | IN | 1 | all | 0.0299089 | -0.000774636 | 0.553333 | 60 | 39 |
| D2 | MATR | IN | 3 | 1001_plus | 0.115504 | -0.00172317 | 0.385714 | 14 | 9 |
| D2 | MATR | IN | 3 | all | 0.0315801 | -0.00244583 | 0.513333 | 60 | 39 |
| D2 | MATR | IN | 5 | 1001_plus | 0.108667 | 0.00511348 | 0.228571 | 14 | 9 |
| D2 | MATR | IN | 5 | all | 0.0293596 | -0.00022535 | 0.533333 | 60 | 39 |
| D2 | MATR | LoRA | 1 | 1001_plus | 0.113517 | 0.000263909 | 0.414286 | 14 | 9 |
| D2 | MATR | LoRA | 1 | all | 0.0291472 | -1.29586e-05 | 0.463333 | 60 | 39 |
| D2 | MATR | LoRA | 3 | 1001_plus | 0.108761 | 0.00501973 | 0.371429 | 14 | 9 |
| D2 | MATR | LoRA | 3 | all | 0.0297616 | -0.000627334 | 0.563333 | 60 | 39 |
| D2 | MATR | LoRA | 5 | 1001_plus | 0.106902 | 0.00687847 | 0.214286 | 14 | 9 |
| D2 | MATR | LoRA | 5 | all | 0.0298925 | -0.000758175 | 0.576667 | 60 | 39 |
| D2 | MATR | LoRA_FA | 1 | 1001_plus | 0.113475 | 0.000305753 | 0.414286 | 14 | 9 |
| D2 | MATR | LoRA_FA | 1 | all | 0.0291353 | -1.02743e-06 | 0.466667 | 60 | 39 |
| D2 | MATR | LoRA_FA | 3 | 1001_plus | 0.107663 | 0.00611799 | 0.371429 | 14 | 9 |
| D2 | MATR | LoRA_FA | 3 | all | 0.0298694 | -0.000735158 | 0.54 | 60 | 39 |
| D2 | MATR | LoRA_FA | 5 | 1001_plus | 0.105123 | 0.00865728 | 0.242857 | 14 | 9 |
| D2 | MATR | LoRA_FA | 5 | all | 0.0308753 | -0.00174097 | 0.586667 | 60 | 39 |
| D2 | MATR | NORMAL | 1 | 1001_plus | 0.115698 | -0.001917 | 0.414286 | 14 | 9 |
| D2 | MATR | NORMAL | 1 | all | 0.0297871 | -0.000652837 | 0.56 | 60 | 39 |
| D2 | MATR | NORMAL | 3 | 1001_plus | 0.110358 | 0.0034222 | 0.342857 | 14 | 9 |
| D2 | MATR | NORMAL | 3 | all | 0.0294731 | -0.000338853 | 0.523333 | 60 | 39 |
| D2 | MATR | NORMAL | 5 | 1001_plus | 0.108947 | 0.00483376 | 0.228571 | 14 | 9 |
| D2 | MATR | NORMAL | 5 | all | 0.0295522 | -0.000417971 | 0.553333 | 60 | 39 |
| D2 | MATR | OUT | 1 | 1001_plus | 0.114867 | -0.00108651 | 0.5 | 14 | 9 |
| D2 | MATR | OUT | 1 | all | 0.0293939 | -0.000259672 | 0.543333 | 60 | 39 |
| D2 | MATR | OUT | 3 | 1001_plus | 0.112373 | 0.00140738 | 0.457143 | 14 | 9 |
| D2 | MATR | OUT | 3 | all | 0.0297273 | -0.000593064 | 0.486667 | 60 | 39 |
| D2 | MATR | OUT | 5 | 1001_plus | 0.110685 | 0.00309518 | 0.328571 | 14 | 9 |
| D2 | MATR | OUT | 5 | all | 0.0301022 | -0.000967902 | 0.546667 | 60 | 39 |
| D2 | MATR | ZERO | 0 | 1001_plus | 0.113781 | 0 | 0 | 14 | 9 |
| D2 | MATR | ZERO | 0 | all | 0.0291343 | 0 | 0 | 60 | 39 |

MATR主任务由原54颗目标池中的4个共同可行精确工况组成，固定8颗Query。每个工况获得K颗Support，聚合信息预算为4K。HUST固定39颗Query，Support预算为跨工况池中的K颗。Stanford仅按供应循环坐标列为附加协议。

## 同Query方法差异

| stage | domain | K | window | reference_method | candidate_method | reference_mae | candidate_mae | candidate_gain_over_reference | physical_query_cells | paired_cell_seed_draw_rows | candidate_improved_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D2 | MATR | 1 | 1001_plus | CORE | OUT | 0.117283 | 0.114867 | 0.00241618 | 14 | 70 | 0.428571 |
| D2 | MATR | 1 | 1001_plus | IN | OUT | 0.113446 | 0.114867 | -0.00142057 | 14 | 70 | 0.442857 |
| D2 | MATR | 1 | 1001_plus | NORMAL | OUT | 0.115698 | 0.114867 | 0.000830491 | 14 | 70 | 0.385714 |
| D2 | MATR | 1 | 1001_plus | CAL2 | OUT | 0.135827 | 0.114867 | 0.0209601 | 14 | 70 | 0.571429 |
| D2 | MATR | 1 | 1001_plus | LoRA | OUT | 0.113517 | 0.114867 | -0.00135042 | 14 | 70 | 0.442857 |
| D2 | MATR | 1 | 1001_plus | LoRA_FA | OUT | 0.113475 | 0.114867 | -0.00139226 | 14 | 70 | 0.485714 |
| D2 | MATR | 1 | all | CORE | OUT | 0.0307674 | 0.0293939 | 0.00137343 | 60 | 300 | 0.656667 |
| D2 | MATR | 1 | all | IN | OUT | 0.0299089 | 0.0293939 | 0.000514964 | 60 | 300 | 0.58 |
| D2 | MATR | 1 | all | NORMAL | OUT | 0.0297871 | 0.0293939 | 0.000393165 | 60 | 300 | 0.613333 |
| D2 | MATR | 1 | all | CAL2 | OUT | 0.036895 | 0.0293939 | 0.00750103 | 60 | 300 | 0.683333 |
| D2 | MATR | 1 | all | LoRA | OUT | 0.0291472 | 0.0293939 | -0.000246714 | 60 | 300 | 0.52 |
| D2 | MATR | 1 | all | LoRA_FA | OUT | 0.0291353 | 0.0293939 | -0.000258645 | 60 | 300 | 0.52 |
| D2 | MATR | 3 | 1001_plus | CORE | OUT | 0.111949 | 0.112373 | -0.000424227 | 14 | 70 | 0.642857 |
| D2 | MATR | 3 | 1001_plus | IN | OUT | 0.115504 | 0.112373 | 0.00313055 | 14 | 70 | 0.457143 |
| D2 | MATR | 3 | 1001_plus | NORMAL | OUT | 0.110358 | 0.112373 | -0.00201482 | 14 | 70 | 0.442857 |
| D2 | MATR | 3 | 1001_plus | CAL2 | OUT | 0.124265 | 0.112373 | 0.0118921 | 14 | 70 | 0.685714 |
| D2 | MATR | 3 | 1001_plus | LoRA | OUT | 0.108761 | 0.112373 | -0.00361235 | 14 | 70 | 0.5 |
| D2 | MATR | 3 | 1001_plus | LoRA_FA | OUT | 0.107663 | 0.112373 | -0.00471061 | 14 | 70 | 0.5 |
| D2 | MATR | 3 | all | CORE | OUT | 0.0318769 | 0.0297273 | 0.00214952 | 60 | 300 | 0.666667 |
| D2 | MATR | 3 | all | IN | OUT | 0.0315801 | 0.0297273 | 0.00185277 | 60 | 300 | 0.573333 |
| D2 | MATR | 3 | all | NORMAL | OUT | 0.0294731 | 0.0297273 | -0.000254212 | 60 | 300 | 0.526667 |
| D2 | MATR | 3 | all | CAL2 | OUT | 0.0344105 | 0.0297273 | 0.00468314 | 60 | 300 | 0.666667 |
| D2 | MATR | 3 | all | LoRA | OUT | 0.0297616 | 0.0297273 | 3.427e-05 | 60 | 300 | 0.616667 |
| D2 | MATR | 3 | all | LoRA_FA | OUT | 0.0298694 | 0.0297273 | 0.000142094 | 60 | 300 | 0.62 |
| D2 | MATR | 5 | 1001_plus | CORE | OUT | 0.111424 | 0.110685 | 0.000738784 | 14 | 70 | 0.657143 |
| D2 | MATR | 5 | 1001_plus | IN | OUT | 0.108667 | 0.110685 | -0.0020183 | 14 | 70 | 0.414286 |
| D2 | MATR | 5 | 1001_plus | NORMAL | OUT | 0.108947 | 0.110685 | -0.00173858 | 14 | 70 | 0.542857 |
| D2 | MATR | 5 | 1001_plus | CAL2 | OUT | 0.119739 | 0.110685 | 0.00905352 | 14 | 70 | 0.642857 |
| D2 | MATR | 5 | 1001_plus | LoRA | OUT | 0.106902 | 0.110685 | -0.00378328 | 14 | 70 | 0.5 |
| D2 | MATR | 5 | 1001_plus | LoRA_FA | OUT | 0.105123 | 0.110685 | -0.0055621 | 14 | 70 | 0.514286 |
| D2 | MATR | 5 | all | CORE | OUT | 0.033132 | 0.0301022 | 0.00302981 | 60 | 300 | 0.696667 |
| D2 | MATR | 5 | all | IN | OUT | 0.0293596 | 0.0301022 | -0.000742551 | 60 | 300 | 0.463333 |
| D2 | MATR | 5 | all | NORMAL | OUT | 0.0295522 | 0.0301022 | -0.000549931 | 60 | 300 | 0.496667 |
| D2 | MATR | 5 | all | CAL2 | OUT | 0.0338713 | 0.0301022 | 0.00376909 | 60 | 300 | 0.71 |
| D2 | MATR | 5 | all | LoRA | OUT | 0.0298925 | 0.0301022 | -0.000209726 | 60 | 300 | 0.646667 |
| D2 | MATR | 5 | all | LoRA_FA | OUT | 0.0308753 | 0.0301022 | 0.000773073 | 60 | 300 | 0.636667 |

candidate_gain_over_reference为逐cell、同seed及draw的参考方法MAE减去候选方法MAE，正值支持候选改善。每行明确当前已评分的配对数量。D3仅使用seed42，三训练种子证据由D2和D4分别提供。

## 抽样与训练种子波动

| stage | domain | method | K | window | seed_std_after_averaging_draws | mean_within_seed_task_draw_std | seeds | draw_variance_includes_explicit_repeated_support_slots | independent_units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D2 | MATR | CAL2 | 1 | all | NA | 0.00452184 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | CAL2 | 3 | all | NA | 0.00410019 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | CAL2 | 5 | all | NA | 0.00360471 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | CORE | 1 | all | NA | 0.00130859 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | CORE | 3 | all | NA | 0.00294605 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | CORE | 5 | all | NA | 0.00241003 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | IN | 1 | all | NA | 0.000812169 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | IN | 3 | all | NA | 0.00283033 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | IN | 5 | all | NA | 0.000762523 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | LoRA | 1 | all | NA | 3.23846e-05 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | LoRA | 3 | all | NA | 0.000598212 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | LoRA | 5 | all | NA | 0.000614488 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | LoRA_FA | 1 | all | NA | 6.44739e-05 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | LoRA_FA | 3 | all | NA | 0.00080053 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | LoRA_FA | 5 | all | NA | 0.00176929 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | NORMAL | 1 | all | NA | 0.000699342 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | NORMAL | 3 | all | NA | 0.000574883 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | NORMAL | 5 | all | NA | 0.000524696 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | OUT | 1 | all | NA | 0.000367728 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | OUT | 3 | all | NA | 0.00146394 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | OUT | 5 | all | NA | 0.00141949 | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | ZERO | 0 | all | NA | NA | 1 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |

draw波动在同task与seed内计算，seed波动先对draw平均后计算。重复Support集合与相同adaptation_id保留，独立样本数按物理电芯与工况计数。

## 适配成本

按唯一adaptation_id计，非线性适配器拟合82次，局部正规化求解205次。ADAPTATION_COSTS.csv保留每个唯一状态的实际优化步数、cache_solver、logtime、字典字节与线性化差距，未测量字段保留NA。

当前共有290条唯一状态回执。源参考模型训练次数在REFERENCE_MODELS.csv单列。

所选范围的累计engine计数包含inner与评价，实际非线性优化更新7020次。已完成源参考模型3组、训练阶段拟合6次，累计源优化更新5760次。不同阶段汇总共享的参考训练只能计一次。

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
