# 单侧方向少样本迁移实验复核

## 研究结论与目标测试

D0至D4已完成。945个源非零适配槽、135个谱对照槽及675个目标非零适配槽全部评分，对应1533个唯一非零状态。12个共同参考模型由24次源训练阶段拟合得到。D3沿用D2的seed42共同K0函数，参考评分重复单列。

表中数值为三个训练seed、五个Support draw在固定Query上的每电池完整已观测未来MAE均值。SOH采用标称容量单位。MATR固定8颗Query，K按每个精确协议任务计，四任务合计使用4K颗完整轨迹Support。HUST固定39颗Query，使用pooled K颗Support。

| 域 | 方法 | K0 | K1 | K3 | K5 | K5相对K0改善 |
| --- | --- | --- | --- | --- | --- | --- |
| MATR | CAL2 | 0.015413 | 0.015209 | 0.014715 | 0.014811 | 3.90% |
| MATR | LoRA | 0.015413 | 0.015380 | 0.014534 | 0.014752 | 4.29% |
| MATR | OUT | 0.015413 | 0.015161 | 0.014519 | 0.014551 | 5.59% |
| HUST | CAL2 | 0.036558 | 0.035428 | 0.029600 | 0.029411 | 19.55% |
| HUST | LoRA | 0.036558 | 0.036548 | 0.033502 | 0.032900 | 10.01% |
| HUST | OUT | 0.036558 | 0.037911 | 0.035140 | 0.034953 | 4.39% |

OUT的MATR K3、K5 MAE分别低于标准LoRA约0.000015、0.000201。HUST K5的OUT MAE为0.034953，LoRA为0.032900，CAL2为0.029411。当前证据支持有限的目标校准收益，跨域稳定的方法优势仍需进一步证据。MATR的1001+窗口只有4颗物理Query，K3、K5的OUT均略差于LoRA。K5远期MAE分别为0.038518与0.037402。完整未来与远期的改善方向应分别解释。MATR完整未来只有8颗物理Query，seed和draw属于重复评价轴。

按完整已观测未来MAE计算，源outer的7种方法在非零K上平均的增益均为负。标准LoRA与LoRA-FA的1-shot改善接近零，其余K曲线与负迁移均保留于原始表。源LoRA与LoRA-FA在1001+窗口跨K平均增益分别约0.002879、0.003456，该窗口有14颗物理Query。源回顾性开发表现与目标校准表现具有明显差异。

D3的CROSS在K1、K3、K5完整未来MAE分别为0.029502、0.031769、0.030434，基线为0.029134。远期三个K同样恶化。CROSS的第三折线性化偏差较大，结论约束当前方向选择和单次局部求解组合。三折AUTO、CROSS、RANDOM使用匹配的7、7、8维，详细分析见stage_snapshots/D3_SCIENTIFIC_BRIEF_ZH.md。

备用接口探针的登记触发条件为假。MATR和HUST均观察到真实目标校准收益，INTERFACE_PROBE_DECISION.json记录完整判据及数字。

## 预算与工程实现

共同Bridge/Head为59585参数，源静态核心为192参数。目标OUT仅更新16个共享系数，标准LoRA更新9216参数，CAL2更新2参数。OUT额外冻结字典为1253376字节，NORMAL为113246208字节。可训练参数与完整存储成本分别报告。

源参考训练累计24285次优化更新。包含inner选参及评价的非线性适配累计26550次实际更新，另有11364次FP64局部求解。只统计正式评价状态的更新为12870次，两个口径分别保留。

RTX4060独立推理基准采用一个seed、固定K3 draw0代表状态、批量8及4900点完整网格，预热1次、同步计时3次。HUST附加分支实现的OUT与LoRA分别约3.18和3.06毫秒每电池，同状态合并实现分别约2.796和2.871毫秒。三次计时只描述这些代表场景。完整重放、合并一致性、显存和逐次时长见INFERENCE_BENCHMARK.csv及JSON。

Jacobian采用torch.autograd.functional.jvp实现。官方模型的两方向FP64有限差分相对误差约1.97e-8。最终137项回归通过，实际适配状态均执行保存重放，几何和LoRA增量均核验合并一致性。

## 证据边界与下一项对照

设计优势为共同起点、固定Query、独立完整轨迹Support和真实非线性评分。主要风险来自历史源域暴露、有限目标电池数、局部线性化误差和Support协议覆盖。HUST按实际已观测轨迹评分。对于本登记矩阵的数值复核，置信度高。对于跨电池域的普适机制优势，置信度低。

目标Query已在本轮评分并登记暴露。用户新增的服务器对照继续复用这些固定电池，将比较共同Bridge/Head下的预训练主干LoRA、源专门化LoRA/LoRA-XS与OUT16，并分别估计各自K0适配增益及配对方法差异。其协议与源内选参独立登记于specialization_comparison_v1。

## 当前执行范围

状态为COMPLETE。已评分1779个请求槽，待完成0个，失败或不可估计0个。实际适配优化更新12870次。

输入合同为VIQ+earlySOH / no-policy。native_multivariate使用每圈12个统计描述量。源模型从policy维度0初始化，SOH锚点及前缀单位来源单列于输入审计。

## 完整未来与远期K曲线

| stage | domain | method | K | window | mae | gain_mae | negative_transfer_ratio | physical_query_cells | protocol_groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D2 | MATR | CAL2 | 1 | 1001_plus | 0.129262 | -0.0156823 | 0.542857 | 14 | 9 |
| D2 | MATR | CAL2 | 1 | all | 0.0354605 | -0.00657508 | 0.548889 | 60 | 39 |
| D2 | MATR | CAL2 | 3 | 1001_plus | 0.124483 | -0.0109032 | 0.485714 | 14 | 9 |
| D2 | MATR | CAL2 | 3 | all | 0.0341026 | -0.00521711 | 0.627778 | 60 | 39 |
| D2 | MATR | CAL2 | 5 | 1001_plus | 0.124649 | -0.0110692 | 0.328571 | 14 | 9 |
| D2 | MATR | CAL2 | 5 | all | 0.0361614 | -0.00727599 | 0.638889 | 60 | 39 |
| D2 | MATR | CORE | 1 | 1001_plus | 0.121768 | -0.00818848 | 0.466667 | 14 | 9 |
| D2 | MATR | CORE | 1 | all | 0.0315657 | -0.00268024 | 0.568889 | 60 | 39 |
| D2 | MATR | CORE | 3 | 1001_plus | 0.12384 | -0.0102606 | 0.428571 | 14 | 9 |
| D2 | MATR | CORE | 3 | all | 0.0342399 | -0.0053544 | 0.56 | 60 | 39 |
| D2 | MATR | CORE | 5 | 1001_plus | 0.116508 | -0.00292851 | 0.328571 | 14 | 9 |
| D2 | MATR | CORE | 5 | all | 0.033125 | -0.00423951 | 0.575556 | 60 | 39 |
| D2 | MATR | IN | 1 | 1001_plus | 0.11639 | -0.00281065 | 0.466667 | 14 | 9 |
| D2 | MATR | IN | 1 | all | 0.0303116 | -0.00142611 | 0.536667 | 60 | 39 |
| D2 | MATR | IN | 3 | 1001_plus | 0.126865 | -0.0132855 | 0.42381 | 14 | 9 |
| D2 | MATR | IN | 3 | all | 0.0342465 | -0.00536107 | 0.506667 | 60 | 39 |
| D2 | MATR | IN | 5 | 1001_plus | 0.122583 | -0.00900336 | 0.285714 | 14 | 9 |
| D2 | MATR | IN | 5 | all | 0.0330129 | -0.00412748 | 0.517778 | 60 | 39 |
| D2 | MATR | LoRA | 1 | 1001_plus | 0.113479 | 0.000100464 | 0.447619 | 14 | 9 |
| D2 | MATR | LoRA | 1 | all | 0.0288793 | 6.12723e-06 | 0.421111 | 60 | 39 |
| D2 | MATR | LoRA | 3 | 1001_plus | 0.110447 | 0.00313231 | 0.4 | 14 | 9 |
| D2 | MATR | LoRA | 3 | all | 0.0306847 | -0.00179929 | 0.563333 | 60 | 39 |
| D2 | MATR | LoRA | 5 | 1001_plus | 0.108174 | 0.00540528 | 0.238095 | 14 | 9 |
| D2 | MATR | LoRA | 5 | all | 0.0296526 | -0.000767117 | 0.556667 | 60 | 39 |
| D2 | MATR | LoRA_FA | 1 | 1001_plus | 0.113559 | 2.03195e-05 | 0.447619 | 14 | 9 |
| D2 | MATR | LoRA_FA | 1 | all | 0.0288788 | 6.69888e-06 | 0.431111 | 60 | 39 |
| D2 | MATR | LoRA_FA | 3 | 1001_plus | 0.109133 | 0.00444608 | 0.380952 | 14 | 9 |
| D2 | MATR | LoRA_FA | 3 | all | 0.0306316 | -0.00174617 | 0.557778 | 60 | 39 |
| D2 | MATR | LoRA_FA | 5 | 1001_plus | 0.107676 | 0.00590303 | 0.261905 | 14 | 9 |
| D2 | MATR | LoRA_FA | 5 | all | 0.0314544 | -0.00256897 | 0.587778 | 60 | 39 |
| D2 | MATR | NORMAL | 1 | 1001_plus | 0.118896 | -0.00531691 | 0.42381 | 14 | 9 |
| D2 | MATR | NORMAL | 1 | all | 0.0304322 | -0.00154672 | 0.536667 | 60 | 39 |
| D2 | MATR | NORMAL | 3 | 1001_plus | 0.113404 | 0.000175153 | 0.380952 | 14 | 9 |
| D2 | MATR | NORMAL | 3 | all | 0.0302208 | -0.00133531 | 0.511111 | 60 | 39 |
| D2 | MATR | NORMAL | 5 | 1001_plus | 0.115946 | -0.00236637 | 0.285714 | 14 | 9 |
| D2 | MATR | NORMAL | 5 | all | 0.0312757 | -0.00239022 | 0.537778 | 60 | 39 |
| D2 | MATR | OUT | 1 | 1001_plus | 0.115364 | -0.00178457 | 0.480952 | 14 | 9 |
| D2 | MATR | OUT | 1 | all | 0.0297151 | -0.000829646 | 0.538889 | 60 | 39 |
| D2 | MATR | OUT | 3 | 1001_plus | 0.114844 | -0.00126407 | 0.452381 | 14 | 9 |
| D2 | MATR | OUT | 3 | all | 0.0300825 | -0.00119704 | 0.495556 | 60 | 39 |
| D2 | MATR | OUT | 5 | 1001_plus | 0.113025 | 0.00055493 | 0.32381 | 14 | 9 |
| D2 | MATR | OUT | 5 | all | 0.0302473 | -0.00136185 | 0.527778 | 60 | 39 |
| D2 | MATR | ZERO | 0 | 1001_plus | 0.113579 | 0 | 0 | 14 | 9 |
| D2 | MATR | ZERO | 0 | all | 0.0288855 | 0 | 0 | 60 | 39 |
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
| D4 | HUST | CAL2 | 1 | 1001_plus | 0.059093 | 0.0010382 | 0.470085 | 39 | 39 |
| D4 | HUST | CAL2 | 1 | all | 0.0354278 | 0.00113045 | 0.459829 | 39 | 39 |
| D4 | HUST | CAL2 | 3 | 1001_plus | 0.0516692 | 0.008462 | 0.17094 | 39 | 39 |
| D4 | HUST | CAL2 | 3 | all | 0.0296002 | 0.00695801 | 0.0803419 | 39 | 39 |
| D4 | HUST | CAL2 | 5 | 1001_plus | 0.0514718 | 0.00865939 | 0.152137 | 39 | 39 |
| D4 | HUST | CAL2 | 5 | all | 0.0294115 | 0.00714676 | 0.0666667 | 39 | 39 |
| D4 | HUST | LoRA | 1 | 1001_plus | 0.0601385 | -7.36538e-06 | 0.45641 | 39 | 39 |
| D4 | HUST | LoRA | 1 | all | 0.0365485 | 9.74785e-06 | 0.449573 | 39 | 39 |
| D4 | HUST | LoRA | 3 | 1001_plus | 0.0546857 | 0.0054455 | 0.398291 | 39 | 39 |
| D4 | HUST | LoRA | 3 | all | 0.0335022 | 0.00305603 | 0.384615 | 39 | 39 |
| D4 | HUST | LoRA | 5 | 1001_plus | 0.0539825 | 0.00614869 | 0.348718 | 39 | 39 |
| D4 | HUST | LoRA | 5 | all | 0.0328996 | 0.00365864 | 0.323077 | 39 | 39 |
| D4 | HUST | OUT | 1 | 1001_plus | 0.0630897 | -0.00295849 | 0.492308 | 39 | 39 |
| D4 | HUST | OUT | 1 | all | 0.0379114 | -0.00135312 | 0.468376 | 39 | 39 |
| D4 | HUST | OUT | 3 | 1001_plus | 0.0578154 | 0.0023158 | 0.441026 | 39 | 39 |
| D4 | HUST | OUT | 3 | all | 0.03514 | 0.00141827 | 0.425641 | 39 | 39 |
| D4 | HUST | OUT | 5 | 1001_plus | 0.0576009 | 0.00253028 | 0.389744 | 39 | 39 |
| D4 | HUST | OUT | 5 | all | 0.0349528 | 0.00160548 | 0.350427 | 39 | 39 |
| D4 | HUST | ZERO | 0 | 1001_plus | 0.0601312 | 0 | 0 | 39 | 39 |
| D4 | HUST | ZERO | 0 | all | 0.0365582 | 0 | 0 | 39 | 39 |
| D4 | MATR | CAL2 | 1 | 1001_plus | 0.0410768 | 0.00159454 | 0.4 | 4 | 4 |
| D4 | MATR | CAL2 | 1 | all | 0.0152093 | 0.000203746 | 0.533333 | 8 | 4 |
| D4 | MATR | CAL2 | 3 | 1001_plus | 0.0381345 | 0.0045368 | 0.35 | 4 | 4 |
| D4 | MATR | CAL2 | 3 | all | 0.0147148 | 0.000698277 | 0.408333 | 8 | 4 |
| D4 | MATR | CAL2 | 5 | 1001_plus | 0.0393633 | 0.00330799 | 0.35 | 4 | 4 |
| D4 | MATR | CAL2 | 5 | all | 0.0148114 | 0.000601656 | 0.325 | 8 | 4 |
| D4 | MATR | LoRA | 1 | 1001_plus | 0.0423965 | 0.000274794 | 0.4 | 4 | 4 |
| D4 | MATR | LoRA | 1 | all | 0.0153795 | 3.35183e-05 | 0.483333 | 8 | 4 |
| D4 | MATR | LoRA | 3 | 1001_plus | 0.0377626 | 0.00490868 | 0.3 | 4 | 4 |
| D4 | MATR | LoRA | 3 | all | 0.0145342 | 0.000878869 | 0.466667 | 8 | 4 |
| D4 | MATR | LoRA | 5 | 1001_plus | 0.0374021 | 0.00526916 | 0.183333 | 4 | 4 |
| D4 | MATR | LoRA | 5 | all | 0.0147517 | 0.000661377 | 0.533333 | 8 | 4 |
| D4 | MATR | OUT | 1 | 1001_plus | 0.0393775 | 0.0032938 | 0.433333 | 4 | 4 |
| D4 | MATR | OUT | 1 | all | 0.0151612 | 0.000251824 | 0.541667 | 8 | 4 |
| D4 | MATR | OUT | 3 | 1001_plus | 0.0378791 | 0.00479216 | 0.316667 | 4 | 4 |
| D4 | MATR | OUT | 3 | all | 0.0145194 | 0.00089365 | 0.458333 | 8 | 4 |
| D4 | MATR | OUT | 5 | 1001_plus | 0.0385177 | 0.00415362 | 0.333333 | 4 | 4 |
| D4 | MATR | OUT | 5 | all | 0.0145509 | 0.000862116 | 0.341667 | 8 | 4 |
| D4 | MATR | ZERO | 0 | 1001_plus | 0.0426713 | 0 | 0 | 4 | 4 |
| D4 | MATR | ZERO | 0 | all | 0.015413 | 0 | 0 | 8 | 4 |

MATR主任务由原54颗目标池中的4个共同可行精确工况组成，固定8颗Query。每个工况获得K颗Support，聚合信息预算为4K。HUST固定39颗Query，Support预算为跨工况池中的K颗。Stanford仅按供应循环坐标列为附加协议。

## 同Query方法差异

| stage | domain | K | window | reference_method | candidate_method | reference_mae | candidate_mae | candidate_gain_over_reference | physical_query_cells | paired_cell_seed_draw_rows | candidate_improved_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D2 | MATR | 1 | 1001_plus | CORE | OUT | 0.121768 | 0.115364 | 0.0064039 | 14 | 210 | 0.438095 |
| D2 | MATR | 1 | 1001_plus | IN | OUT | 0.11639 | 0.115364 | 0.00102607 | 14 | 210 | 0.447619 |
| D2 | MATR | 1 | 1001_plus | NORMAL | OUT | 0.118896 | 0.115364 | 0.00353234 | 14 | 210 | 0.414286 |
| D2 | MATR | 1 | 1001_plus | CAL2 | OUT | 0.129262 | 0.115364 | 0.0138977 | 14 | 210 | 0.57619 |
| D2 | MATR | 1 | 1001_plus | LoRA | OUT | 0.113479 | 0.115364 | -0.00188504 | 14 | 210 | 0.519048 |
| D2 | MATR | 1 | 1001_plus | LoRA_FA | OUT | 0.113559 | 0.115364 | -0.00180489 | 14 | 210 | 0.547619 |
| D2 | MATR | 1 | all | CORE | OUT | 0.0315657 | 0.0297151 | 0.00185059 | 60 | 900 | 0.622222 |
| D2 | MATR | 1 | all | IN | OUT | 0.0303116 | 0.0297151 | 0.000596467 | 60 | 900 | 0.586667 |
| D2 | MATR | 1 | all | NORMAL | OUT | 0.0304322 | 0.0297151 | 0.000717075 | 60 | 900 | 0.554444 |
| D2 | MATR | 1 | all | CAL2 | OUT | 0.0354605 | 0.0297151 | 0.00574544 | 60 | 900 | 0.614444 |
| D2 | MATR | 1 | all | LoRA | OUT | 0.0288793 | 0.0297151 | -0.000835773 | 60 | 900 | 0.457778 |
| D2 | MATR | 1 | all | LoRA_FA | OUT | 0.0288788 | 0.0297151 | -0.000836345 | 60 | 900 | 0.482222 |
| D2 | MATR | 3 | 1001_plus | CORE | OUT | 0.12384 | 0.114844 | 0.00899657 | 14 | 210 | 0.595238 |
| D2 | MATR | 3 | 1001_plus | IN | OUT | 0.126865 | 0.114844 | 0.0120215 | 14 | 210 | 0.485714 |
| D2 | MATR | 3 | 1001_plus | NORMAL | OUT | 0.113404 | 0.114844 | -0.00143922 | 14 | 210 | 0.47619 |
| D2 | MATR | 3 | 1001_plus | CAL2 | OUT | 0.124483 | 0.114844 | 0.00963915 | 14 | 210 | 0.671429 |
| D2 | MATR | 3 | 1001_plus | LoRA | OUT | 0.110447 | 0.114844 | -0.00439638 | 14 | 210 | 0.52381 |
| D2 | MATR | 3 | 1001_plus | LoRA_FA | OUT | 0.109133 | 0.114844 | -0.00571015 | 14 | 210 | 0.52381 |
| D2 | MATR | 3 | all | CORE | OUT | 0.0342399 | 0.0300825 | 0.00415736 | 60 | 900 | 0.65 |
| D2 | MATR | 3 | all | IN | OUT | 0.0342465 | 0.0300825 | 0.00416403 | 60 | 900 | 0.581111 |
| D2 | MATR | 3 | all | NORMAL | OUT | 0.0302208 | 0.0300825 | 0.000138264 | 60 | 900 | 0.532222 |
| D2 | MATR | 3 | all | CAL2 | OUT | 0.0341026 | 0.0300825 | 0.00402006 | 60 | 900 | 0.685556 |
| D2 | MATR | 3 | all | LoRA | OUT | 0.0306847 | 0.0300825 | 0.00060225 | 60 | 900 | 0.623333 |
| D2 | MATR | 3 | all | LoRA_FA | OUT | 0.0306316 | 0.0300825 | 0.000549126 | 60 | 900 | 0.636667 |
| D2 | MATR | 5 | 1001_plus | CORE | OUT | 0.116508 | 0.113025 | 0.00348344 | 14 | 210 | 0.590476 |
| D2 | MATR | 5 | 1001_plus | IN | OUT | 0.122583 | 0.113025 | 0.00955829 | 14 | 210 | 0.490476 |
| D2 | MATR | 5 | 1001_plus | NORMAL | OUT | 0.115946 | 0.113025 | 0.0029213 | 14 | 210 | 0.566667 |
| D2 | MATR | 5 | 1001_plus | CAL2 | OUT | 0.124649 | 0.113025 | 0.0116241 | 14 | 210 | 0.590476 |
| D2 | MATR | 5 | 1001_plus | LoRA | OUT | 0.108174 | 0.113025 | -0.00485035 | 14 | 210 | 0.514286 |
| D2 | MATR | 5 | 1001_plus | LoRA_FA | OUT | 0.107676 | 0.113025 | -0.0053481 | 14 | 210 | 0.547619 |
| D2 | MATR | 5 | all | CORE | OUT | 0.033125 | 0.0302473 | 0.00287765 | 60 | 900 | 0.637778 |
| D2 | MATR | 5 | all | IN | OUT | 0.0330129 | 0.0302473 | 0.00276562 | 60 | 900 | 0.535556 |
| D2 | MATR | 5 | all | NORMAL | OUT | 0.0312757 | 0.0302473 | 0.00102837 | 60 | 900 | 0.515556 |
| D2 | MATR | 5 | all | CAL2 | OUT | 0.0361614 | 0.0302473 | 0.00591414 | 60 | 900 | 0.772222 |
| D2 | MATR | 5 | all | LoRA | OUT | 0.0296526 | 0.0302473 | -0.000594736 | 60 | 900 | 0.634444 |
| D2 | MATR | 5 | all | LoRA_FA | OUT | 0.0314544 | 0.0302473 | 0.00120712 | 60 | 900 | 0.675556 |
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
| D4 | HUST | 1 | 1001_plus | CAL2 | OUT | 0.059093 | 0.0630897 | -0.00399669 | 39 | 585 | 0.509402 |
| D4 | HUST | 1 | 1001_plus | LoRA | OUT | 0.0601385 | 0.0630897 | -0.00295113 | 39 | 585 | 0.504274 |
| D4 | HUST | 1 | all | CAL2 | OUT | 0.0354278 | 0.0379114 | -0.00248357 | 39 | 585 | 0.505983 |
| D4 | HUST | 1 | all | LoRA | OUT | 0.0365485 | 0.0379114 | -0.00136287 | 39 | 585 | 0.526496 |
| D4 | HUST | 3 | 1001_plus | CAL2 | OUT | 0.0516692 | 0.0578154 | -0.00614621 | 39 | 585 | 0.193162 |
| D4 | HUST | 3 | 1001_plus | LoRA | OUT | 0.0546857 | 0.0578154 | -0.0031297 | 39 | 585 | 0.422222 |
| D4 | HUST | 3 | all | CAL2 | OUT | 0.0296002 | 0.03514 | -0.00553973 | 39 | 585 | 0.0923077 |
| D4 | HUST | 3 | all | LoRA | OUT | 0.0335022 | 0.03514 | -0.00163776 | 39 | 585 | 0.42735 |
| D4 | HUST | 5 | 1001_plus | CAL2 | OUT | 0.0514718 | 0.0576009 | -0.00612911 | 39 | 585 | 0.181197 |
| D4 | HUST | 5 | 1001_plus | LoRA | OUT | 0.0539825 | 0.0576009 | -0.00361841 | 39 | 585 | 0.389744 |
| D4 | HUST | 5 | all | CAL2 | OUT | 0.0294115 | 0.0349528 | -0.00554129 | 39 | 585 | 0.0820513 |
| D4 | HUST | 5 | all | LoRA | OUT | 0.0328996 | 0.0349528 | -0.00205316 | 39 | 585 | 0.365812 |
| D4 | MATR | 1 | 1001_plus | CAL2 | OUT | 0.0410768 | 0.0393775 | 0.00169925 | 4 | 60 | 0.633333 |
| D4 | MATR | 1 | 1001_plus | LoRA | OUT | 0.0423965 | 0.0393775 | 0.003019 | 4 | 60 | 0.666667 |
| D4 | MATR | 1 | all | CAL2 | OUT | 0.0152093 | 0.0151612 | 4.80777e-05 | 8 | 120 | 0.525 |
| D4 | MATR | 1 | all | LoRA | OUT | 0.0153795 | 0.0151612 | 0.000218305 | 8 | 120 | 0.466667 |
| D4 | MATR | 3 | 1001_plus | CAL2 | OUT | 0.0381345 | 0.0378791 | 0.000255369 | 4 | 60 | 0.6 |
| D4 | MATR | 3 | 1001_plus | LoRA | OUT | 0.0377626 | 0.0378791 | -0.000116514 | 4 | 60 | 0.566667 |
| D4 | MATR | 3 | all | CAL2 | OUT | 0.0147148 | 0.0145194 | 0.000195373 | 8 | 120 | 0.466667 |
| D4 | MATR | 3 | all | LoRA | OUT | 0.0145342 | 0.0145194 | 1.47806e-05 | 8 | 120 | 0.5 |
| D4 | MATR | 5 | 1001_plus | CAL2 | OUT | 0.0393633 | 0.0385177 | 0.000845624 | 4 | 60 | 0.65 |
| D4 | MATR | 5 | 1001_plus | LoRA | OUT | 0.0374021 | 0.0385177 | -0.00111555 | 4 | 60 | 0.35 |
| D4 | MATR | 5 | all | CAL2 | OUT | 0.0148114 | 0.0145509 | 0.00026046 | 8 | 120 | 0.55 |
| D4 | MATR | 5 | all | LoRA | OUT | 0.0147517 | 0.0145509 | 0.000200739 | 8 | 120 | 0.641667 |

candidate_gain_over_reference为逐cell、同seed及draw的参考方法MAE减去候选方法MAE，正值支持候选改善。每行明确当前已评分的配对数量。D3仅使用seed42，三训练种子证据由D2和D4分别提供。

## 抽样与训练种子波动

| stage | domain | method | K | window | seed_std_after_averaging_draws | mean_within_seed_task_draw_std | seeds | draw_variance_includes_explicit_repeated_support_slots | independent_units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D2 | MATR | CAL2 | 1 | all | 0.00341972 | 0.00537997 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | CAL2 | 3 | all | 0.000362605 | 0.00382177 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | CAL2 | 5 | all | 0.00199779 | 0.0033681 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | CORE | 1 | all | 0.000792779 | 0.00252906 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | CORE | 3 | all | 0.00291046 | 0.00447611 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | CORE | 5 | all | 0.00127006 | 0.00259133 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | IN | 1 | all | 0.000353654 | 0.00142046 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | IN | 3 | all | 0.00231203 | 0.00547872 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | IN | 5 | all | 0.00321368 | 0.00388535 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | LoRA | 1 | all | 0.000292935 | 3.74935e-05 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | LoRA | 3 | all | 0.00105176 | 0.00223769 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | LoRA | 5 | all | 0.000208423 | 0.000454597 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | LoRA_FA | 1 | all | 0.000278153 | 9.42951e-05 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | LoRA_FA | 3 | all | 0.000712816 | 0.0020119 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | LoRA_FA | 5 | all | 0.001585 | 0.00149837 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | NORMAL | 1 | all | 0.000584111 | 0.00164817 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | NORMAL | 3 | all | 0.000647545 | 0.00115317 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | NORMAL | 5 | all | 0.00216905 | 0.00124989 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | OUT | 1 | all | 0.00032256 | 0.000896523 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | OUT | 3 | all | 0.000581681 | 0.00204191 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | OUT | 5 | all | 0.000540694 | 0.00178072 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D2 | MATR | ZERO | 0 | all | 0.000280757 | NA | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
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
| D4 | HUST | CAL2 | 1 | all | 0.00181524 | 0.00163867 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | HUST | CAL2 | 3 | all | 0.000900487 | 0.0018117 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | HUST | CAL2 | 5 | all | 0.000938303 | 0.00153826 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | HUST | LoRA | 1 | all | 0.00187226 | 0.000265427 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | HUST | LoRA | 3 | all | 0.00240059 | 0.00185634 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | HUST | LoRA | 5 | all | 0.00198201 | 0.00188688 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | HUST | OUT | 1 | all | 0.00269437 | 0.00414539 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | HUST | OUT | 3 | all | 0.00246012 | 0.00137964 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | HUST | OUT | 5 | all | 0.00225559 | 0.00126907 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | HUST | ZERO | 0 | all | 0.00193755 | NA | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | MATR | CAL2 | 1 | all | 0.00165731 | 0.000875761 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | MATR | CAL2 | 3 | all | 0.0014211 | 0.000596165 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | MATR | CAL2 | 5 | all | 0.00149979 | 0.000362981 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | MATR | LoRA | 1 | all | 0.00180128 | 0.000259443 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | MATR | LoRA | 3 | all | 0.00157839 | 0.000875126 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | MATR | LoRA | 5 | all | 0.00165751 | 0.000635087 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | MATR | OUT | 1 | all | 0.00165989 | 0.00121937 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | MATR | OUT | 3 | all | 0.00148708 | 0.000537555 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | MATR | OUT | 5 | all | 0.00153253 | 0.000379741 | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |
| D4 | MATR | ZERO | 0 | all | 0.00181102 | NA | 3 | True | physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes |

draw波动在同task与seed内计算，seed波动先对draw平均后计算。重复Support集合与相同adaptation_id保留，独立样本数按物理电芯与工况计数。

## 适配成本

按唯一adaptation_id计，非线性适配器拟合429次，局部正规化求解1104次。ADAPTATION_COSTS.csv保留每个唯一状态的实际优化步数、cache_solver、logtime、字典字节与线性化差距，未测量字段保留NA。

当前共有1560条唯一状态回执。源参考模型训练次数在REFERENCE_MODELS.csv单列。

所选范围的累计engine计数包含inner与评价，实际非线性优化更新26550次。已完成源参考模型12组、训练阶段拟合24次，累计源优化更新24285次。不同阶段汇总共享的参考训练只能计一次。

## 失败与不可估计

登记矩阵中的失败或不可估计记录为0。

远期窗口没有观测时保留NA。Support下降、Query恶化作为负迁移保留，源开发负结果继续进入已锁定的目标评价流程。

## 优势与风险

优势为共同f0、固定Query、嵌套物理Support、fold-local构基及真实非线性前向评分。四种几何臂与通用LoRA基线分别登记维度、冻结存储和优化成本。

风险包括源115颗与三折的历史暴露、训练集重叠、有限远期支持、局部线性近似误差，以及Support协议覆盖不足。source空间方向不赋予未验证的电化学含义。

现有数字的解释以实际覆盖为限。汇总程序只读取已评分表，未读取目标未来标签数组，也未新增bootstrap或模型运行。数值汇总核验置信度取决于请求槽覆盖与算术一致性。

## 拓展思考

结合K曲线、最强通用基线、CAL2、负迁移与成本判断下一步，单独的梯度捕获比例不作为方法成功依据。

## 你可能想知道

输入或超参数选择只使用源内部inner任务。目标一经评分登记为已暴露，当前结果不能重新作为未见目标测试。
