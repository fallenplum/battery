# E1–E3 工况组交叉验证结果

## 完成状态与总判定

**本轮总判定为 NOT_SUPPORTED_SOURCE_E123，目标校准推进条件为 False。**

| candidate | E1_supported | E2_supported | E3_supported | TARGET_ADVANCE_ALLOWED |
| --- | --- | --- | --- | --- |
| absolute | False | False | False | False |
| dual | False | False | False | False |

absolute与dual在训练前登记为两个主要候选。全部E1、E2、E3判据必须由同一个候选满足，才允许推进正式1/3/5-shot目标校准及原封存跨域评价。本报告只评价源内部外层工况。

## 数据、训练与评价范围

源池共115颗物理电芯、62个exact protocol groups，三外层折、三个训练种子。外层电芯数为39、38、38，1001+电芯数为7、7、7。合并OOF完整未来支持115颗、62组，1001+支持21颗、13组。所有源电芯在每种配置和种子下只外层评价一次。

登记训练单元为216，其中warmup9、静态核心27、条件化180，正式优化更新预算207360。全部387个外层结果完成后才运行本汇总。模型、超参数、幅度及结构在外层评分前冻结，选权仅用每折inner-validation。最终MATR目标54颗及跨域Query的未来标签使用量为0。

历史源池已参与前期研究，本轮属于retrospective嵌套group CV。各折fit有部分重叠，不能将三个折或三个种子视为九个完全独立数据集。归一化、梯度基、Bridge/Head与静态核心均按本折fit/inner重建。

## 主幅度结果

| 子空间参数化 | 视角 | 幅度评价 | 1001+ MAE | 完整未来MAE |
| --- | --- | --- | --- | --- |
| grad_core | absolute | matched_norm | 0.095834 | 0.033021 |
| grad_core | absolute | trained_ball | 0.100473 | 0.025718 |
| grad_core | constant | matched_norm | 0.097646 | 0.033528 |
| grad_core | constant | trained_ball | 0.100533 | 0.025744 |
| grad_core | dual | matched_norm | 0.097665 | 0.033083 |
| grad_core | dual | trained_ball | 0.100464 | 0.025701 |
| grad_core | relative | matched_norm | 0.096961 | 0.033008 |
| grad_core | relative | trained_ball | 0.100669 | 0.025773 |
| lora_xs | absolute | matched_norm | 0.104877 | 0.025370 |
| lora_xs | absolute | trained_ball | 0.104762 | 0.025451 |
| lora_xs | constant | matched_norm | 0.104699 | 0.025386 |
| lora_xs | constant | trained_ball | 0.104728 | 0.025462 |
| lora_xs | dual | matched_norm | 0.104955 | 0.025385 |
| lora_xs | dual | trained_ball | 0.104785 | 0.025461 |
| lora_xs | relative | matched_norm | 0.104769 | 0.025366 |
| lora_xs | relative | trained_ball | 0.104767 | 0.025460 |
| weight_svd_core | absolute | matched_norm | 0.106724 | 0.025396 |
| weight_svd_core | absolute | trained_ball | 0.105106 | 0.025521 |
| weight_svd_core | constant | matched_norm | 0.106664 | 0.025400 |
| weight_svd_core | constant | trained_ball | 0.105040 | 0.025526 |
| weight_svd_core | dual | matched_norm | 0.107375 | 0.025437 |
| weight_svd_core | dual | trained_ball | 0.105283 | 0.025528 |
| weight_svd_core | relative | matched_norm | 0.106734 | 0.025404 |
| weight_svd_core | relative | trained_ball | 0.105088 | 0.025527 |

每种条件化方法仅训练192个delta-core和1248个门参数，共1440。Bridge、Head及静态C0冻结。所有视角从同一参数化的同一C0和单位门开始。绝对与相对视角的标准化仅使用fit，固定电压网格使各视角四个V统计恒定，参数数目相同仍存在有效函数自由度差异。

trained_ball限制每cell×layer的物理权重增量，matched_norm对同一已选checkpoint的非零增量进行固定范数评价。增量以静态C0为起点，包含核心和门的联合效应。radius1对应每层官方W0 Frobenius范数的1%，Gradient-Core额外训练radius0.25与4。

## E1 状态条件化与幅度控制

| 候选 | 幅度评价 | 窗口 | 改善百分比 | 电芯CI下界 | 电芯CI上界 | 工况CI下界 | 工况CI上界 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| absolute | trained_ball | all | 0.101089 | -0.000157 | 0.000195 | -0.000135 | 0.000119 |
| absolute | trained_ball | 1001_plus | 0.059378 | -0.001059 | 0.001355 | -0.001561 | 0.001958 |
| dual | trained_ball | all | 0.167132 | -0.000116 | 0.000184 | -0.000087 | 0.000196 |
| dual | trained_ball | 1001_plus | 0.068323 | -0.001457 | 0.002007 | -0.001806 | 0.002778 |
| absolute | matched_norm | all | 1.513296 | -0.000319 | 0.001921 | -0.000208 | 0.001316 |
| absolute | matched_norm | 1001_plus | 1.855451 | -0.003861 | 0.010609 | -0.004425 | 0.014877 |
| dual | matched_norm | all | 1.328330 | -0.000429 | 0.001463 | -0.000699 | 0.001197 |
| dual | matched_norm | 1001_plus | -0.020290 | -0.008171 | 0.011366 | -0.010336 | 0.016973 |

区间对象为常数门MAE减去候选MAE，正值对应候选改善。主点估计采用电芯等权，工况区间先组内平均再组等权。置信水平为97.5%，两个主要候选预登记，各区间使用2000次训练seed与电芯或工况的交叉重抽样。区间属于固定训练设置下的描述，保留历史暴露和训练重叠的解释边界。

E1要求同一候选在Gradient-Core主幅度、两种评价方式均改善至少1%，三个fold及三个seed边际方向均正，两类区间下界均正，额外两个幅度的完整未来方向均正。九个fold×seed的逐项方向保留在E123_CONTRASTS.json。

### 三个更新半径的完整未来方向

| 候选 | 幅度评价 | R | 完整未来改善% | 折0改善% | 折1改善% | 折2改善% |
| --- | --- | --- | --- | --- | --- | --- |
| absolute | trained_ball | 0.250000 | -0.150456 | -1.022132 | 0.669721 | -0.033837 |
| absolute | trained_ball | 1.000000 | 0.101089 | -0.221170 | 0.513891 | 0.059525 |
| absolute | trained_ball | 4.000000 | 0.598135 | 0.468415 | 1.187291 | 0.242190 |
| dual | trained_ball | 0.250000 | -0.315925 | -1.323266 | 0.524133 | -0.094544 |
| dual | trained_ball | 1.000000 | 0.167132 | -0.145215 | 0.576700 | 0.119246 |
| dual | trained_ball | 4.000000 | 0.549576 | 0.543847 | 0.806627 | 0.348292 |
| absolute | matched_norm | 0.250000 | -2.629793 | -7.880393 | 0.553234 | -0.456294 |
| absolute | matched_norm | 1.000000 | 1.513296 | -1.147451 | 0.595195 | 6.279402 |
| absolute | matched_norm | 4.000000 | -2.835783 | -8.486177 | 0.768057 | 3.692776 |
| dual | matched_norm | 0.250000 | -2.767880 | -8.041217 | 0.972310 | -1.018398 |
| dual | matched_norm | 1.000000 | 1.328330 | -0.376603 | -2.291544 | 6.643613 |
| dual | matched_norm | 4.000000 | -11.332049 | -10.901921 | -27.261935 | 2.759215 |

表中改善均以对应半径、方向来源和幅度评价的constant为参照。正值为候选MAE下降，负值完整保留。三个半径均预先登记，评价结果没有用于重新选择幅度。

此前D3双视角相对静态模型的完整未来平均改善为8.19%。本轮Gradient-Core主半径trained_ball的dual相对本轮静态C0，完整未来MAE增加0.409%。本轮同时改变了Bridge/Head冻结方式、fit电芯规模、工况留出与幅度控制条件。原D3的改善幅度没有在这些联合条件下复现。当前设计尚未分别隔离这些变化的贡献，单一因素的因果解释仍需对应对照。

## E2 子空间来源与收益等效

| 候选 | 幅度评价 | 梯度减权重来源收益差 | 电芯CI下界 | 电芯CI上界 | 工况CI下界 | 工况CI上界 | 等效判据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| absolute | trained_ball | 0.000822 | -0.005801 | 0.006530 | -0.005710 | 0.005645 | True |
| dual | trained_ball | 0.001761 | -0.005328 | 0.007855 | -0.004214 | 0.009557 | True |
| absolute | matched_norm | 0.014978 | -0.007399 | 0.063862 | -0.005359 | 0.055169 | False |
| dual | matched_norm | 0.014747 | -0.012098 | 0.043032 | -0.022115 | 0.039742 | False |

收益差定义为Gradient-Core的相对改善减去orthogonal weight-SVD核心的相对改善，每种改善由同次重抽样中constant和候选MAE的均值比计算。两类97.5%区间均完全落在±0.02内才满足等效判据，额外要求三参数化、两种评价方式均有正完整未来收益。

LoRA-XS与orthogonal weight-SVD拥有相同的权重SVD U/V空间，XS吸收Σ作为坐标缩放。三分支覆盖梯度与权重两类方向来源，并包含XS参数化对照。相关收益不能解释为三个独立方向空间的验证。

| 候选 | 方向参数化 | 幅度评价 | 常数门MAE | 候选MAE | 完整未来改善% |
| --- | --- | --- | --- | --- | --- |
| absolute | grad_core | trained_ball | 0.025744 | 0.025718 | 0.101089 |
| absolute | lora_xs | trained_ball | 0.025462 | 0.025451 | 0.043053 |
| absolute | weight_svd_core | trained_ball | 0.025526 | 0.025521 | 0.018849 |
| dual | grad_core | trained_ball | 0.025744 | 0.025701 | 0.167132 |
| dual | lora_xs | trained_ball | 0.025462 | 0.025461 | 0.004939 |
| dual | weight_svd_core | trained_ball | 0.025526 | 0.025528 | -0.009008 |
| absolute | grad_core | matched_norm | 0.033528 | 0.033021 | 1.513296 |
| absolute | lora_xs | matched_norm | 0.025386 | 0.025370 | 0.061380 |
| absolute | weight_svd_core | matched_norm | 0.025400 | 0.025396 | 0.015473 |
| dual | grad_core | matched_norm | 0.033528 | 0.033083 | 1.328330 |
| dual | lora_xs | matched_norm | 0.025386 | 0.025385 | 0.003319 |
| dual | weight_svd_core | matched_norm | 0.025400 | 0.025437 | -0.146336 |

trained_ball下主要候选与三种参数化的完整未来改善范围为-0.0090%至0.1671%。该幅度评价中满足收益差等效的候选为absolute、dual。等效判据描述两类来源的增量收益差范围，机制有效性需要同时结合收益幅度、工况方向与E1/E3证据。这些trained_ball增量收益均未达到登记的1%平均改善幅度，当前等效结果发生在接近零的收益上，状态条件化机制的有效性尚未建立。matched_norm下未满足收益等效的主要候选为absolute、dual。

## E3 独立工况方向一致性

E3分别检查Gradient-Core与orthogonal weight-SVD，两种幅度评价下每个外层fold的完整未来与1001+方向均需为正。合并1001+改善与两类区间下界也均需为正。每fold×seed、每组和全部分段结果保存于CSV和逐项gate JSON，负方向保留。

合并OOF支持115颗完整未来电芯与21颗1001+电芯，下表各折数值先跨本折电芯与三个seed平均。物理支持数按唯一电芯计数。

### 主半径 trained_ball

| 方向来源 | 候选 | 折 | 完整未来改善% | 1001+改善% | 完整/远期电芯 | 完整/远期工况 |
| --- | --- | --- | --- | --- | --- | --- |
| grad_core | absolute | 0 | -0.221170 | 0.701487 | 39/7 | 21/5 |
| grad_core | absolute | 1 | 0.513891 | 0.372487 | 38/7 | 21/4 |
| grad_core | absolute | 2 | 0.059525 | -0.909622 | 38/7 | 20/4 |
| weight_svd_core | absolute | 0 | -0.064188 | 0.003588 | 39/7 | 21/5 |
| weight_svd_core | absolute | 1 | 0.241455 | -0.136994 | 38/7 | 21/4 |
| weight_svd_core | absolute | 2 | -0.085873 | -0.111836 | 38/7 | 20/4 |
| grad_core | dual | 0 | -0.145215 | 1.083869 | 39/7 | 21/5 |
| grad_core | dual | 1 | 0.576700 | -0.626093 | 38/7 | 21/4 |
| grad_core | dual | 2 | 0.119246 | -0.864087 | 38/7 | 20/4 |
| weight_svd_core | dual | 0 | -0.031198 | -0.284499 | 39/7 | 21/5 |
| weight_svd_core | dual | 1 | 0.217105 | -0.112550 | 38/7 | 21/4 |
| weight_svd_core | dual | 2 | -0.169963 | -0.221176 | 38/7 | 20/4 |

### 主半径 matched_norm

| 方向来源 | 候选 | 折 | 完整未来改善% | 1001+改善% | 完整/远期电芯 | 完整/远期工况 |
| --- | --- | --- | --- | --- | --- | --- |
| grad_core | absolute | 0 | -1.147451 | 0.765754 | 39/7 | 21/5 |
| grad_core | absolute | 1 | 0.595195 | 2.438159 | 38/7 | 21/4 |
| grad_core | absolute | 2 | 6.279402 | 2.763840 | 38/7 | 20/4 |
| weight_svd_core | absolute | 0 | -0.095692 | -0.062666 | 39/7 | 21/5 |
| weight_svd_core | absolute | 1 | 0.155430 | -0.144340 | 38/7 | 21/4 |
| weight_svd_core | absolute | 2 | 0.000333 | -0.000617 | 38/7 | 20/4 |
| grad_core | dual | 0 | -0.376603 | 1.163140 | 39/7 | 21/5 |
| grad_core | dual | 1 | -2.291544 | -9.130500 | 38/7 | 21/4 |
| grad_core | dual | 2 | 6.643613 | 3.050905 | 38/7 | 20/4 |
| weight_svd_core | dual | 0 | -0.353702 | -1.224670 | 39/7 | 21/5 |
| weight_svd_core | dual | 1 | -0.101374 | -0.404101 | 38/7 | 21/4 |
| weight_svd_core | dual | 2 | -0.000259 | -0.001096 | 38/7 | 20/4 |

两种幅度评价分别保留每折完整未来与1001+的正负方向，任何一个预登记方向为负都会体现于原E3判据。该呈现只展开既有逐折收益字段。

## 独立核验

重新从私有NPZ计算74175条电芯窗口MAE/RMSE/bias，最大数值差1e-16。原始监督及观察mask与冻结缓存逐值匹配。物理增量控制通过状态为True，matched-norm不可达到的cell×layer数为0。

若与候选相关配置的某个增量为零，matched-norm无法获得指定半径，该行完整保留并使对应实验的幅度隔离支持判定失败。局部判定按候选、方向来源与实验范围执行，其count、分布与有效覆盖保存在norm审计和gate JSON。计算所有已登记外层结果后统一统计，不以结果选半径或选候选。

## 优势、风险与置信度

优势是工况组隔离、折内来源重建、固定适配器起点、相同可训练参数量以及物理权重空间幅度控制，可以共同约束参数容量与优化尺度解释。风险包括115颗源数据历史暴露、各折训练重叠、仅21颗远期支持、恒定视角维和不同参数化的训练动力学。

执行与核验完整性的置信度高，依据完整训练闭环、387个外层结果、独立NPZ重算与来源摘要。物理因果与外域泛化的解释取决于本轮全部gate支持情况和后续封存目标评价。

## 拓展思考

E2若支持两类方向来源获得等效收益，后续机制定位应围绕状态条件化，并保留梯度构基作为来源选择对照。E1/E3的失败项需要与其幅度和工况分布一起保留。

## 你可能想知道

本汇总执行0次训练和0次优化更新。最终目标Query未来轨迹在本轮保持封存，目标推进取决于同一主要候选同时满足三个登记判据。
