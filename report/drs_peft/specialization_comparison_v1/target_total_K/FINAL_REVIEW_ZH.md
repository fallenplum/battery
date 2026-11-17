# 源域专门化与目标少样本适配复核

## 当前实验结论

**两域均按全域总K执行，K为0、1、3、5。372个请求槽已全部评分，包含360个非零适配槽和12个family ZERO参考槽。** 三个模型种子为42、123、2026，每个非零K保留五次登记的Support抽样。MATR从24颗候选中总共选K颗，统一适配后评价8颗固定Query。HUST从38颗候选中总共选K颗，评价39颗固定Query。两域使用相同的物理Support身份与Query身份进行方法间配对。

### 完整已观测未来的主结果

MAE基于归一化容量轨迹，先按每颗Query的真实观测点计算，再按物理电芯、种子与登记draw汇总。表中Δ5为各方法自己的K0减去K5，正值表示目标适配改善。

| 数据集 | 方法 | 目标可训练参数 | K0 | K1 | K3 | K5 | Δ5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MATR | 预训练主干 + LoRA | 9216 | 0.016023 | 0.016069 | 0.016290 | 0.016441 | -0.000417 |
| MATR | 源专门化主干 + LoRA | 9216 | 0.015413 | 0.015540 | 0.016155 | 0.016139 | -0.000726 |
| MATR | 源专门化主干 + XS | 192 | 0.015413 | 0.015445 | 0.015437 | 0.015489 | -0.000076 |
| MATR | 源专门化主干 + OUT16 | 16 | 0.015413 | 0.016028 | 0.015942 | 0.015761 | -0.000348 |
| HUST | 预训练主干 + LoRA | 9216 | 0.035917 | 0.035904 | 0.032251 | 0.032048 | +0.003870 |
| HUST | 源专门化主干 + LoRA | 9216 | 0.036558 | 0.036548 | 0.033502 | 0.032900 | +0.003659 |
| HUST | 源专门化主干 + XS | 192 | 0.036558 | 0.036547 | 0.036526 | 0.036453 | +0.000105 |
| HUST | 源专门化主干 + OUT16 | 16 | 0.036558 | 0.037911 | 0.035140 | 0.034953 | +0.001605 |

- **MATR的少样本适配未改善完整轨迹平均MAE。** 四种方法在K1、K3、K5的汇总ΔK均为负。源专门化的LoRA、XS与OUT在每个K、每个种子的五draw平均完整轨迹ΔK均为负。最低汇总MAE来自源专门化K0，为0.015413。已适配方法中XS最接近该起点。
- **HUST在K3和K5能利用少量Support改善预测。** 在本轮四种适配方法中，预训练主干加LoRA的完整轨迹平均误差最低。其K5相对自身K0改善10.774%，专门化LoRA改善10.008%，OUT改善4.392%，XS改善0.288%。OUT在K1发生负迁移。
- **OUT16的稳定独立预测优势尚未得到支持。** MATR的OUT在K3和K5优于专门化LoRA，同时仍劣于自身K0和XS。HUST的OUT在K3和K5优于XS，同时劣于两种LoRA。参数预算差异与求解方式差异均保留在比较解释中。

### OUT与同源专门化参考上的适配器配对

下表为OUT的MAE减去比较方法MAE。负值支持OUT，正值支持比较方法。

| 数据集 | K | OUT − LoRA | OUT − XS |
| --- | ---: | ---: | ---: |
| MATR | 1 | +0.000487 | +0.000582 |
| MATR | 3 | -0.000213 | +0.000504 |
| MATR | 5 | -0.000378 | +0.000273 |
| HUST | 1 | +0.001363 | +0.001364 |
| HUST | 3 | +0.001638 | -0.001386 |
| HUST | 5 | +0.002053 | -0.001500 |

### 远期与种子边界

MATR的1001圈以后窗口只有4颗有观测Query。OUT的K5远期MAE从0.042671改善至0.042339，Δ为+0.000333，同时完整轨迹Δ为-0.000348。远期正向结果受观测覆盖与种子限制，K3只有1个种子的五draw平均Δ为正，K5有2个。不同窗口的改善方向需要共同报告。

HUST的1001圈以后窗口覆盖39颗Query。K5的预训练LoRA、专门化LoRA、XS和OUT远期MAE依次为0.053654、0.053982、0.059932和0.057601。OUT相对自身远期K0的Δ为+0.002530，两种LoRA的远期误差均更低。

HUST四种方法在K3、K5的完整轨迹与远期ΔK，按每个种子的五draw平均后均为正。抽样层面仍有负迁移，例如OUT在K5有13/15个seed/draw记录改善，预训练LoRA有15/15个。三种子五draw属于重复评价轴，独立物理Query数量维持39颗。

源专门化对K0的影响具有明显种子依赖。MATR的SPECIALIZED_ZERO减去WARMUP_ZERO完整轨迹MAE，按42、123、2026顺序分别为+0.000603、-0.003286、+0.000852。HUST依次为-0.001448、+0.004722、-0.001351。MATR三种子均值的起点优势、HUST三种子均值的起点劣势主要受到seed123较大位移影响。两种LoRA最终绝对MAE的差异须结合各自K0与ΔK解释。

HUST在K3和K5的完整与远期窗口，专门化LoRA分别在2/3个种子上取得低于预训练LoRA的MAE，同时预训练LoRA的三种子平均更低。OUT在上述四种K与窗口组合中，三个种子的五draw平均MAE均高于专门化LoRA。独立复算的140组均值、130组配对差异及种子方向见INDEPENDENT_REVIEW.json。

### 监督利用与机制诊断

360个非零请求记录的Support MSE全部下降，同时MATR有130/180个记录出现Query电池平均MAE增加。HUST K5的预训练LoRA、专门化LoRA、OUT和XS分别有15/15、14/15、13/15、11/15个seed/draw记录改善。按请求槽计数包含重复Support，独立物理Query电池仍为MATR 8颗、HUST 39颗。

XS的84个不同适配状态均有非零有效更新，其Σ与专门化权重对应核验通过。MATR K5的XS更新范数约为OUT的1.47倍。XS较小的平均收益具有真实更新背景，方向敏感性、优化设置与信息限制的因果贡献仍需独立证据。OUT的首Query网格线性化偏差随域、种子和Support变化。逐组数值、冻结因子存储开销及诊断范围见[适配诊断复核](ADAPTATION_DIAGNOSTICS_ZH.md)。

### 输入、实现选择与信息预算

所有组使用前100圈VIQ及earlySOH，包含共同的59585参数源训练Bridge/Head。源专门化采用已有12层rank4静态梯度核心，共192参数。目标适配冻结Bridge、Head、官方主干参数及既有源核心。这里的专门化范围对应该静态核心实现。

每颗Support提供其已观测未来轨迹的监督，每次适配最多确定性选取512个真实观测点，损失按物理电芯等权。Query仅在实际观测点评分，预测坐标统一为101至5000。HUST含右删失记录，已观测轨迹长度随电芯变化。

XS使用专门化有效权重的完整FP64 SVD并保留Σ。为固定共同起点，本轮采用零增量初始化，方法身份为LoRA-XS zero-increment variant。LoRA、XS与OUT分别使用9216、192、16个目标可训练参数。四种方法的参数与优化预算逐项记录，当前结果支持实际效果比较，其机制归因受多项结构差异限制。

MATR的K1五draw仅对应3个不同Support集合，K3和K5各有5个。重复draw按登记保留。Query沿用历史已经暴露的固定池，本轮评价属于同Query配对控制比较。源inner选参与目标锁均在本轮评分前完成。

### 执行与交付核验

源inner完成18个作业、3240个候选评分槽，实际新增13680次优化更新。目标完成372个请求槽，对应336个唯一非零状态，其中246个为新增状态。新目标实际新增6210次优化更新。HUST精确复用93个父实验槽，其中90个非零适配槽、3个ZERO参考槽。MATR当前总K适配全部使用新pooled身份。

新增source与target均在服务器RTX A6000的7号、6号GPU执行，每卡一个本研究worker。HUST精确复用部分来自既有RTX4060运行。XS新source和新target分别进行108次、36次矩阵SVD。实际优化、缓存命中、选定快照步数和历史复用成本分列。

服务器回传5772个文件，归档SHA256与全部文件SHA256、CRC已核验。本地完整评分后，独立交付核验确认372槽覆盖、Support与Query分离、每次Support数量严格等于K、预测网格和状态hash一致。回归测试57项通过。可追溯文件为DELIVERY_VERIFICATION.json、REGRESSION_SEAL.json、EXECUTION_LEDGER.json及逐槽回执。

### 优势、风险与置信度

**优势**在于共同Bridge/Head、相同物理Support与Query、独立family ZERO、完整与远期共同报告，以及状态和评分事实的完整留存。OUT使用16维可训练预算，额外冻结因子的存储为1253376字节，XS对应147648字节。可训练参数与冻结因子共同计量适配开销。

**风险**包括目标Query历史暴露、MATR仅8颗Query且远期仅4颗、重复Support抽样、三种子对专门化均值的影响、参数与求解预算差异，以及Support拟合与Query泛化之间的偏离。数值结果与本轮覆盖的核验置信度为高。向新Query、新工况推广当前排序的置信度为低。当前证据对跨域稳定的OUT独立优势支持不足。

## 逐项结果与自动汇总

## 覆盖与输入

状态为COMPLETE。已评分372槽，待完成0槽，失败或不可估计0槽。

输入为VIQ+earlySOH / no-policy。两类参考共享源训练的59585参数Bridge/Head，SPECIALIZED额外包含192参数冻结源核心。warmup参考使用官方预训练主干。目标Query属于已有历史暴露的固定池。

## 同Query的K曲线

| domain | method | reference_family | K | window | mae | zero_mae | delta_K | negative_transfer_ratio | physical_query_cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HUST | PRETRAINED_LORA | WARMUP | 1 | 1001_plus | 0.059933 | 0.0599231 | -9.90622e-06 | 0.434188 | 39 |
| HUST | PRETRAINED_LORA | WARMUP | 1 | all | 0.0359044 | 0.0359173 | 1.29525e-05 | 0.430769 | 39 |
| HUST | PRETRAINED_LORA | WARMUP | 3 | 1001_plus | 0.0536132 | 0.0599231 | 0.00630986 | 0.355556 | 39 |
| HUST | PRETRAINED_LORA | WARMUP | 3 | all | 0.0322512 | 0.0359173 | 0.00366607 | 0.347009 | 39 |
| HUST | PRETRAINED_LORA | WARMUP | 5 | 1001_plus | 0.0536541 | 0.0599231 | 0.00626896 | 0.321368 | 39 |
| HUST | PRETRAINED_LORA | WARMUP | 5 | all | 0.0320476 | 0.0359173 | 0.00386974 | 0.297436 | 39 |
| HUST | SPECIALIZED_LORA | SPECIALIZED | 1 | 1001_plus | 0.0601385 | 0.0601312 | -7.36538e-06 | 0.45641 | 39 |
| HUST | SPECIALIZED_LORA | SPECIALIZED | 1 | all | 0.0365485 | 0.0365582 | 9.74785e-06 | 0.449573 | 39 |
| HUST | SPECIALIZED_LORA | SPECIALIZED | 3 | 1001_plus | 0.0546857 | 0.0601312 | 0.0054455 | 0.398291 | 39 |
| HUST | SPECIALIZED_LORA | SPECIALIZED | 3 | all | 0.0335022 | 0.0365582 | 0.00305603 | 0.384615 | 39 |
| HUST | SPECIALIZED_LORA | SPECIALIZED | 5 | 1001_plus | 0.0539825 | 0.0601312 | 0.00614869 | 0.348718 | 39 |
| HUST | SPECIALIZED_LORA | SPECIALIZED | 5 | all | 0.0328996 | 0.0365582 | 0.00365864 | 0.323077 | 39 |
| HUST | SPECIALIZED_OUT | SPECIALIZED | 1 | 1001_plus | 0.0630897 | 0.0601312 | -0.00295849 | 0.492308 | 39 |
| HUST | SPECIALIZED_OUT | SPECIALIZED | 1 | all | 0.0379114 | 0.0365582 | -0.00135312 | 0.468376 | 39 |
| HUST | SPECIALIZED_OUT | SPECIALIZED | 3 | 1001_plus | 0.0578154 | 0.0601312 | 0.0023158 | 0.441026 | 39 |
| HUST | SPECIALIZED_OUT | SPECIALIZED | 3 | all | 0.03514 | 0.0365582 | 0.00141827 | 0.425641 | 39 |
| HUST | SPECIALIZED_OUT | SPECIALIZED | 5 | 1001_plus | 0.0576009 | 0.0601312 | 0.00253028 | 0.389744 | 39 |
| HUST | SPECIALIZED_OUT | SPECIALIZED | 5 | all | 0.0349528 | 0.0365582 | 0.00160548 | 0.350427 | 39 |
| HUST | SPECIALIZED_XS | SPECIALIZED | 1 | 1001_plus | 0.0601187 | 0.0601312 | 1.24738e-05 | 0.444444 | 39 |
| HUST | SPECIALIZED_XS | SPECIALIZED | 1 | all | 0.0365475 | 0.0365582 | 1.07704e-05 | 0.432479 | 39 |
| HUST | SPECIALIZED_XS | SPECIALIZED | 3 | 1001_plus | 0.0600682 | 0.0601312 | 6.2933e-05 | 0.490598 | 39 |
| HUST | SPECIALIZED_XS | SPECIALIZED | 3 | all | 0.0365258 | 0.0365582 | 3.24333e-05 | 0.492308 | 39 |
| HUST | SPECIALIZED_XS | SPECIALIZED | 5 | 1001_plus | 0.0599324 | 0.0601312 | 0.000198783 | 0.494017 | 39 |
| HUST | SPECIALIZED_XS | SPECIALIZED | 5 | all | 0.0364528 | 0.0365582 | 0.000105393 | 0.500855 | 39 |
| HUST | SPECIALIZED_ZERO | SPECIALIZED | 0 | 1001_plus | 0.0601312 | 0.0601312 | 0 | 0 | 39 |
| HUST | SPECIALIZED_ZERO | SPECIALIZED | 0 | all | 0.0365582 | 0.0365582 | 0 | 0 | 39 |
| HUST | WARMUP_ZERO | WARMUP | 0 | 1001_plus | 0.0599231 | 0.0599231 | 0 | 0 | 39 |
| HUST | WARMUP_ZERO | WARMUP | 0 | all | 0.0359173 | 0.0359173 | 0 | 0 | 39 |
| MATR | PRETRAINED_LORA | WARMUP | 1 | 1001_plus | 0.0433797 | 0.0435868 | 0.000207091 | 0.416667 | 4 |
| MATR | PRETRAINED_LORA | WARMUP | 1 | all | 0.0160687 | 0.0160234 | -4.52938e-05 | 0.45 | 8 |
| MATR | PRETRAINED_LORA | WARMUP | 3 | 1001_plus | 0.0437871 | 0.0435868 | -0.000200246 | 0.433333 | 4 |
| MATR | PRETRAINED_LORA | WARMUP | 3 | all | 0.0162897 | 0.0160234 | -0.000266317 | 0.475 | 8 |
| MATR | PRETRAINED_LORA | WARMUP | 5 | 1001_plus | 0.0429157 | 0.0435868 | 0.000671115 | 0.35 | 4 |
| MATR | PRETRAINED_LORA | WARMUP | 5 | all | 0.0164406 | 0.0160234 | -0.000417246 | 0.516667 | 8 |
| MATR | SPECIALIZED_LORA | SPECIALIZED | 1 | 1001_plus | 0.0424807 | 0.0426713 | 0.000190574 | 0.35 | 4 |
| MATR | SPECIALIZED_LORA | SPECIALIZED | 1 | all | 0.0155405 | 0.015413 | -0.000127471 | 0.65 | 8 |
| MATR | SPECIALIZED_LORA | SPECIALIZED | 3 | 1001_plus | 0.0435243 | 0.0426713 | -0.000853051 | 0.4 | 4 |
| MATR | SPECIALIZED_LORA | SPECIALIZED | 3 | all | 0.0161552 | 0.015413 | -0.000742144 | 0.666667 | 8 |
| MATR | SPECIALIZED_LORA | SPECIALIZED | 5 | 1001_plus | 0.042868 | 0.0426713 | -0.000196731 | 0.3 | 4 |
| MATR | SPECIALIZED_LORA | SPECIALIZED | 5 | all | 0.016139 | 0.015413 | -0.000725954 | 0.6 | 8 |
| MATR | SPECIALIZED_OUT | SPECIALIZED | 1 | 1001_plus | 0.04227 | 0.0426713 | 0.000401282 | 0.35 | 4 |
| MATR | SPECIALIZED_OUT | SPECIALIZED | 1 | all | 0.0160276 | 0.015413 | -0.00061456 | 0.666667 | 8 |
| MATR | SPECIALIZED_OUT | SPECIALIZED | 3 | 1001_plus | 0.0423679 | 0.0426713 | 0.000303407 | 0.383333 | 4 |
| MATR | SPECIALIZED_OUT | SPECIALIZED | 3 | all | 0.0159417 | 0.015413 | -0.000528708 | 0.633333 | 8 |
| MATR | SPECIALIZED_OUT | SPECIALIZED | 5 | 1001_plus | 0.0423387 | 0.0426713 | 0.00033261 | 0.383333 | 4 |
| MATR | SPECIALIZED_OUT | SPECIALIZED | 5 | all | 0.0157615 | 0.015413 | -0.000348425 | 0.675 | 8 |
| MATR | SPECIALIZED_XS | SPECIALIZED | 1 | 1001_plus | 0.0426087 | 0.0426713 | 6.25978e-05 | 0.35 | 4 |
| MATR | SPECIALIZED_XS | SPECIALIZED | 1 | all | 0.0154452 | 0.015413 | -3.21821e-05 | 0.65 | 8 |
| MATR | SPECIALIZED_XS | SPECIALIZED | 3 | 1001_plus | 0.0426109 | 0.0426713 | 6.03959e-05 | 0.383333 | 4 |
| MATR | SPECIALIZED_XS | SPECIALIZED | 3 | all | 0.0154373 | 0.015413 | -2.42792e-05 | 0.616667 | 8 |
| MATR | SPECIALIZED_XS | SPECIALIZED | 5 | 1001_plus | 0.0425209 | 0.0426713 | 0.00015041 | 0.35 | 4 |
| MATR | SPECIALIZED_XS | SPECIALIZED | 5 | all | 0.0154886 | 0.015413 | -7.5541e-05 | 0.65 | 8 |
| MATR | SPECIALIZED_ZERO | SPECIALIZED | 0 | 1001_plus | 0.0426713 | 0.0426713 | 0 | 0 | 4 |
| MATR | SPECIALIZED_ZERO | SPECIALIZED | 0 | all | 0.015413 | 0.015413 | 0 | 0 | 8 |
| MATR | WARMUP_ZERO | WARMUP | 0 | 1001_plus | 0.0435868 | 0.0435868 | 0 | 0 | 4 |
| MATR | WARMUP_ZERO | WARMUP | 0 | all | 0.0160234 | 0.0160234 | 0 | 0 | 8 |

delta_K为各方法自己的family ZERO MAE减去适配MAE，正值表示改善。MATR从24颗候选Support中取全域总共K颗，为固定8颗Query使用同一适配状态。HUST从38颗候选池取全域总K颗，固定39颗Query。

## 配对方法差异

| domain | K | window | left_method | right_method | left_mae | right_mae | difference_mae | delta_K_difference | left_improved_fraction | physical_query_cells | paired_cell_seed_draw_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HUST | 0 | 1001_plus | SPECIALIZED_ZERO | WARMUP_ZERO | 0.0601312 | 0.0599231 | 0.000208127 | 0 | 0.581197 | 39 | 117 |
| HUST | 0 | all | SPECIALIZED_ZERO | WARMUP_ZERO | 0.0365582 | 0.0359173 | 0.000640929 | 0 | 0.555556 | 39 | 117 |
| HUST | 1 | 1001_plus | SPECIALIZED_OUT | SPECIALIZED_LORA | 0.0630897 | 0.0601385 | 0.00295113 | -0.00295113 | 0.504274 | 39 | 585 |
| HUST | 1 | 1001_plus | SPECIALIZED_OUT | SPECIALIZED_XS | 0.0630897 | 0.0601187 | 0.00297097 | -0.00297097 | 0.504274 | 39 | 585 |
| HUST | 1 | 1001_plus | SPECIALIZED_OUT | PRETRAINED_LORA | 0.0630897 | 0.059933 | 0.00315671 | -0.00294859 | 0.524786 | 39 | 585 |
| HUST | 1 | 1001_plus | SPECIALIZED_LORA | PRETRAINED_LORA | 0.0601385 | 0.059933 | 0.000205586 | 2.54084e-06 | 0.582906 | 39 | 585 |
| HUST | 1 | all | SPECIALIZED_OUT | SPECIALIZED_LORA | 0.0379114 | 0.0365485 | 0.00136287 | -0.00136287 | 0.526496 | 39 | 585 |
| HUST | 1 | all | SPECIALIZED_OUT | SPECIALIZED_XS | 0.0379114 | 0.0365475 | 0.00136389 | -0.00136389 | 0.519658 | 39 | 585 |
| HUST | 1 | all | SPECIALIZED_OUT | PRETRAINED_LORA | 0.0379114 | 0.0359044 | 0.002007 | -0.00136607 | 0.519658 | 39 | 585 |
| HUST | 1 | all | SPECIALIZED_LORA | PRETRAINED_LORA | 0.0365485 | 0.0359044 | 0.000644134 | -3.20466e-06 | 0.555556 | 39 | 585 |
| HUST | 3 | 1001_plus | SPECIALIZED_OUT | SPECIALIZED_LORA | 0.0578154 | 0.0546857 | 0.0031297 | -0.0031297 | 0.422222 | 39 | 585 |
| HUST | 3 | 1001_plus | SPECIALIZED_OUT | SPECIALIZED_XS | 0.0578154 | 0.0600682 | -0.00225286 | 0.00225286 | 0.564103 | 39 | 585 |
| HUST | 3 | 1001_plus | SPECIALIZED_OUT | PRETRAINED_LORA | 0.0578154 | 0.0536132 | 0.00420219 | -0.00399406 | 0.461538 | 39 | 585 |
| HUST | 3 | 1001_plus | SPECIALIZED_LORA | PRETRAINED_LORA | 0.0546857 | 0.0536132 | 0.00107248 | -0.000864357 | 0.545299 | 39 | 585 |
| HUST | 3 | all | SPECIALIZED_OUT | SPECIALIZED_LORA | 0.03514 | 0.0335022 | 0.00163776 | -0.00163776 | 0.42735 | 39 | 585 |
| HUST | 3 | all | SPECIALIZED_OUT | SPECIALIZED_XS | 0.03514 | 0.0365258 | -0.00138584 | 0.00138584 | 0.581197 | 39 | 585 |
| HUST | 3 | all | SPECIALIZED_OUT | PRETRAINED_LORA | 0.03514 | 0.0322512 | 0.00288872 | -0.00224779 | 0.42906 | 39 | 585 |
| HUST | 3 | all | SPECIALIZED_LORA | PRETRAINED_LORA | 0.0335022 | 0.0322512 | 0.00125097 | -0.000610037 | 0.512821 | 39 | 585 |
| HUST | 5 | 1001_plus | SPECIALIZED_OUT | SPECIALIZED_LORA | 0.0576009 | 0.0539825 | 0.00361841 | -0.00361841 | 0.389744 | 39 | 585 |
| HUST | 5 | 1001_plus | SPECIALIZED_OUT | SPECIALIZED_XS | 0.0576009 | 0.0599324 | -0.0023315 | 0.0023315 | 0.634188 | 39 | 585 |
| HUST | 5 | 1001_plus | SPECIALIZED_OUT | PRETRAINED_LORA | 0.0576009 | 0.0536541 | 0.0039468 | -0.00373867 | 0.48547 | 39 | 585 |
| HUST | 5 | 1001_plus | SPECIALIZED_LORA | PRETRAINED_LORA | 0.0539825 | 0.0536541 | 0.000328392 | -0.000120266 | 0.567521 | 39 | 585 |
| HUST | 5 | all | SPECIALIZED_OUT | SPECIALIZED_LORA | 0.0349528 | 0.0328996 | 0.00205316 | -0.00205316 | 0.365812 | 39 | 585 |
| HUST | 5 | all | SPECIALIZED_OUT | SPECIALIZED_XS | 0.0349528 | 0.0364528 | -0.00150008 | 0.00150008 | 0.675214 | 39 | 585 |
| HUST | 5 | all | SPECIALIZED_OUT | PRETRAINED_LORA | 0.0349528 | 0.0320476 | 0.00290519 | -0.00226426 | 0.42735 | 39 | 585 |
| HUST | 5 | all | SPECIALIZED_LORA | PRETRAINED_LORA | 0.0328996 | 0.0320476 | 0.000852032 | -0.000211102 | 0.538462 | 39 | 585 |
| MATR | 0 | 1001_plus | SPECIALIZED_ZERO | WARMUP_ZERO | 0.0426713 | 0.0435868 | -0.000915508 | 0 | 0.416667 | 4 | 12 |
| MATR | 0 | all | SPECIALIZED_ZERO | WARMUP_ZERO | 0.015413 | 0.0160234 | -0.000610366 | 0 | 0.708333 | 8 | 24 |
| MATR | 1 | 1001_plus | SPECIALIZED_OUT | SPECIALIZED_LORA | 0.04227 | 0.0424807 | -0.000210708 | 0.000210708 | 0.65 | 4 | 60 |
| MATR | 1 | 1001_plus | SPECIALIZED_OUT | SPECIALIZED_XS | 0.04227 | 0.0426087 | -0.000338684 | 0.000338684 | 0.65 | 4 | 60 |
| MATR | 1 | 1001_plus | SPECIALIZED_OUT | PRETRAINED_LORA | 0.04227 | 0.0433797 | -0.0011097 | 0.000194191 | 0.666667 | 4 | 60 |
| MATR | 1 | 1001_plus | SPECIALIZED_LORA | PRETRAINED_LORA | 0.0424807 | 0.0433797 | -0.000898991 | -1.65169e-05 | 0.416667 | 4 | 60 |
| MATR | 1 | all | SPECIALIZED_OUT | SPECIALIZED_LORA | 0.0160276 | 0.0155405 | 0.000487088 | -0.000487088 | 0.325 | 8 | 120 |
| MATR | 1 | all | SPECIALIZED_OUT | SPECIALIZED_XS | 0.0160276 | 0.0154452 | 0.000582377 | -0.000582377 | 0.333333 | 8 | 120 |
| MATR | 1 | all | SPECIALIZED_OUT | PRETRAINED_LORA | 0.0160276 | 0.0160687 | -4.11e-05 | -0.000569266 | 0.666667 | 8 | 120 |
| MATR | 1 | all | SPECIALIZED_LORA | PRETRAINED_LORA | 0.0155405 | 0.0160687 | -0.000528188 | -8.21777e-05 | 0.683333 | 8 | 120 |
| MATR | 3 | 1001_plus | SPECIALIZED_OUT | SPECIALIZED_LORA | 0.0423679 | 0.0435243 | -0.00115646 | 0.00115646 | 0.566667 | 4 | 60 |
| MATR | 3 | 1001_plus | SPECIALIZED_OUT | SPECIALIZED_XS | 0.0423679 | 0.0426109 | -0.000243011 | 0.000243011 | 0.616667 | 4 | 60 |
| MATR | 3 | 1001_plus | SPECIALIZED_OUT | PRETRAINED_LORA | 0.0423679 | 0.0437871 | -0.00141916 | 0.000503652 | 0.45 | 4 | 60 |
| MATR | 3 | 1001_plus | SPECIALIZED_LORA | PRETRAINED_LORA | 0.0435243 | 0.0437871 | -0.000262703 | -0.000652805 | 0.466667 | 4 | 60 |
| MATR | 3 | all | SPECIALIZED_OUT | SPECIALIZED_LORA | 0.0159417 | 0.0161552 | -0.000213435 | 0.000213435 | 0.558333 | 8 | 120 |
| MATR | 3 | all | SPECIALIZED_OUT | SPECIALIZED_XS | 0.0159417 | 0.0154373 | 0.000504429 | -0.000504429 | 0.383333 | 8 | 120 |
| MATR | 3 | all | SPECIALIZED_OUT | PRETRAINED_LORA | 0.0159417 | 0.0162897 | -0.000347974 | -0.000262392 | 0.633333 | 8 | 120 |
| MATR | 3 | all | SPECIALIZED_LORA | PRETRAINED_LORA | 0.0161552 | 0.0162897 | -0.000134539 | -0.000475827 | 0.608333 | 8 | 120 |
| MATR | 5 | 1001_plus | SPECIALIZED_OUT | SPECIALIZED_LORA | 0.0423387 | 0.042868 | -0.000529342 | 0.000529342 | 0.466667 | 4 | 60 |
| MATR | 5 | 1001_plus | SPECIALIZED_OUT | SPECIALIZED_XS | 0.0423387 | 0.0425209 | -0.0001822 | 0.0001822 | 0.6 | 4 | 60 |
| MATR | 5 | 1001_plus | SPECIALIZED_OUT | PRETRAINED_LORA | 0.0423387 | 0.0429157 | -0.000577003 | -0.000338505 | 0.416667 | 4 | 60 |
| MATR | 5 | 1001_plus | SPECIALIZED_LORA | PRETRAINED_LORA | 0.042868 | 0.0429157 | -4.76614e-05 | -0.000867847 | 0.433333 | 4 | 60 |
| MATR | 5 | all | SPECIALIZED_OUT | SPECIALIZED_LORA | 0.0157615 | 0.016139 | -0.000377529 | 0.000377529 | 0.525 | 8 | 120 |
| MATR | 5 | all | SPECIALIZED_OUT | SPECIALIZED_XS | 0.0157615 | 0.0154886 | 0.000272884 | -0.000272884 | 0.375 | 8 | 120 |
| MATR | 5 | all | SPECIALIZED_OUT | PRETRAINED_LORA | 0.0157615 | 0.0164406 | -0.000679187 | 6.88208e-05 | 0.691667 | 8 | 120 |
| MATR | 5 | all | SPECIALIZED_LORA | PRETRAINED_LORA | 0.016139 | 0.0164406 | -0.000301658 | -0.000308708 | 0.675 | 8 | 120 |

difference_mae为left减去right，负值支持left。delta_K_difference为两方法各自family内增益之差。配对使用相同task、seed、K、draw和物理电芯，远期无观测保留NA。

## 优势与风险

适配预算与每状态成本保存在ADAPTER_BUDGETS.csv，RUN_LEDGER.json的execution_ledger保留collect产生的inner与评价完整执行成本。新增目标成本覆盖MATR全部四种适配方法，以及HUST的PRETRAINED_LORA和SPECIALIZED_XS。HUST父结果的精确复用记录保留其原成本身份。

共同Bridge/Head、固定Query与物理Support，以及独立family ZERO保留了基线位移与目标校准贡献。LoRA为9216参数，XS为192参数，OUT为16个全局共享坐标，预算与求解方式分别报告。

风险包括目标历史暴露、源inner历史使用、有限远期观测及抽样波动。XS采用专门化有效权重的SVD、保留Σ、零增量核心，方法身份为LoRA-XS zero-increment variant。数值核验置信度以实际覆盖为据，方法推广判断还需结合三seed与跨域一致性。

## 拓展思考

结合WARMUP_ZERO与SPECIALIZED_ZERO的差异，以及各自Delta K，可以检验源专门化对迁移起点和Support利用的不同影响。

## 你可能想知道

本汇总只读取公开已评分表与回执，新增模型推理、未来标签数组读取和bootstrap均为0。
