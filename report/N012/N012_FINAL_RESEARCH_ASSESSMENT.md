# N012 最终研究评估

本轮登记日期为2026-11-16，沿用用户直接要求的seed42。全部MATR169和HUST77分别进行单域五折实验。每组150轮、batch8、宽度64、AdamW学习率0.0003、weight decay0.01、梯度裁剪1.0。损失为完整实际观测后缀逐电池等权MAE，选权遵循本折内部验证MAE的strict first-min。所有曲线保留原始150轮，源数据和划分在训练前封存。

MATR拥有123,431个未来观测，HUST拥有138,422个。全部未来观测均保持原始周期与mask，窗口支持不足时记录缺失。SOH使用1.1Ah名义容量归一化，SOH100锚点采用真实96–100圈有效容量中位数。每域分别拟合训练电池的有效点标准化。两域的物理电压窗口和策略维数保持各自定义。

所有队列均有历史研究暴露，本轮属于内部回顾性验证。单种子无法估计训练种子方差。bootstrap区间为当前拟合预测上的条件区间，进行了多个预登记探索性比较。HUST77个精确协议各只有一颗电池，协议重采样缺少组内重复支持。

## 全轨迹主要结果

| domain | model | cell_mean_mae | cell_mean_rmse | protocol_macro_mae | supported_cells | observations |
| --- | --- | --- | --- | --- | --- | --- |
| HUST | N1:nested | 0.023898472 | 0.032261011 | 0.023898472 | 77 | 138422 |
| HUST | N2:free_rate | 0.022357852 | 0.030762474 | 0.022357852 | 77 | 138422 |
| HUST | N2:monotone_spline | 0.020499566 | 0.028370422 | 0.020499566 | 77 | 138422 |
| HUST | N2:nested | 0.021823651 | 0.029530853 | 0.021823651 | 77 | 138422 |
| HUST | N2:physics_basis | 0.021837201 | 0.029912052 | 0.021837201 | 77 | 138422 |
| MATR | HISTORICAL:M1 | 0.018761964 | 0.031578546 | 0.016988782 | 169 | 123431 |
| MATR | N1:nested | 0.030660366 | 0.051671412 | 0.030226917 | 169 | 123431 |
| MATR | N2:free_rate | 0.030537582 | 0.051591350 | 0.029462027 | 169 | 123431 |
| MATR | N2:monotone_spline | 0.030074836 | 0.049478148 | 0.030146012 | 169 | 123431 |
| MATR | N2:nested | 0.030331802 | 0.050474194 | 0.030739376 | 169 | 123431 |
| MATR | N2:physics_basis | 0.030426144 | 0.051521260 | 0.029961871 | 169 | 123431 |

MATR的N1与N2嵌套选模误差均高于历史M1，两种重采样单位的条件区间均支持这一负向差异。历史M1拥有额外显式物理统计、观测残差、边界及速率辅助损失，该差异仍包含信息预算和训练目标的影响。

| candidate | sampling_unit | improvement | ci95_low | ci95_high | better_cells | worse_cells |
| --- | --- | --- | --- | --- | --- | --- |
| N1:nested | cell | -0.011898402 | -0.014627287 | -0.009270512 | 27 | 142 |
| N1:nested | protocol_group | -0.013238135 | -0.016351503 | -0.010121767 | 27 | 142 |
| N2:nested | cell | -0.011569838 | -0.014387349 | -0.008883404 | 37 | 132 |
| N2:nested | protocol_group | -0.013750593 | -0.016833946 | -0.010830644 | 37 | 132 |
## 实际完成与来源

| domain | unique_fits | updates | training_validation_seconds |
| --- | --- | --- | --- |
| HUST | 45 | 47250 | 1467.030 |
| MATR | 45 | 94500 | 1573.866 |

两域共90组真实训练、13,500个正式epochs、141,750次正式优化器更新。额外真实烟测28次更新，合成单测夹具合计78次更新，均单列为工程证据。总执行更新为141856次。合成数据不参与科研成绩。

实际检查通过2953项，90个选定权重的正常入口重放最大绝对误差为0。数据、来源、150轮预算、优化器、验证first-min、全部后缀mask、逐电池汇总、配对区间与独立NumPy解码公式均核验。正式单元按各自encoder/decoder/fold只有一个seed42拟合，断点恢复确定性由合成中断回归检查。中间每轮权重没有单独保存，最终恢复状态、选定权重及全部日志可追溯。

训练前源码提交为`f80ee8f65b751d798bf59c9d5b08dd97bf1a71ed`，测试前评价计划提交为`3fa83014909783bd547fae8faea8a956b481e23f`。原交接ZIP SHA256为`9ae686a8c00aa938dc74f878fa743e911b8216db7f38001e6f310642e8000101`。交接包中的合成测试与无GPU声明依据原STATUS快照解释，本机真实执行以新完成标记为准。

运行环境记录于ENVIRONMENT.json，Python为3.11.14，PyTorch为2.4.1+cu121，编译CUDA为12.1，GPU为NVIDIA GeForce RTX 4060 Laptop GPU, 560.94, 8188 MiB。正式训练串行执行。各单元计时覆盖训练与验证阶段，数据加载和折内标准化在计时器之前执行。

## 三分量基函数的受控证据

MATR相对free_rate的逐电池MAE改善为0.000111438，条件区间尚未一致支持正向增益。

MATR相对monotone_spline的逐电池MAE改善为-0.000351308，条件区间尚未一致支持正向增益。

HUST相对free_rate的逐电池MAE改善为0.000520651，条件区间尚未一致支持正向增益。

HUST相对monotone_spline的逐电池MAE改善为-0.001337635，条件区间尚未一致支持正向增益。

两域中physics_basis相对free_rate与monotone_spline的MAE配对区间均覆盖零，当前三项具体基函数的独立预测增益证据不足。两域中physics_basis的逐电池误差点估计均高于monotone_spline。其非负非增的数学性质已通过公式和逐点检查。当前实验缺少电极分辨测量、真实分量标签和机制可辨识性分析，形状参数化的解释范围限定于这些函数类。

## 后期增益及反例集中度

| domain | window | reference | cells | mean_improvement | better_cells | worse_cells | top3_fraction_of_positive_improvement | largest_harm_cell |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HUST | 1001_plus | free_rate | 77 | -0.000497741 | 41 | 36 | 0.241076063 | HUST_6-6 |
| HUST | 1001_plus | monotone_spline | 77 | -0.002350419 | 33 | 44 | 0.339362070 | HUST_6-6 |
| HUST | 501_1000 | free_rate | 77 | 0.000956669 | 41 | 36 | 0.275758213 | HUST_6-6 |
| HUST | 501_1000 | monotone_spline | 77 | -0.001251106 | 30 | 47 | 0.276394789 | HUST_6-6 |
| MATR | 1001_plus | free_rate | 40 | 0.003534205 | 18 | 22 | 0.635452573 | MATR_b4c5 |
| MATR | 1001_plus | monotone_spline | 40 | -0.010907399 | 15 | 25 | 0.674853207 | MATR_b3c0 |
| MATR | 501_1000 | free_rate | 151 | 0.000277635 | 86 | 65 | 0.170219559 | MATR_b3c6 |
| MATR | 501_1000 | monotone_spline | 151 | 0.000597134 | 67 | 84 | 0.228258291 | MATR_b1c39 |

分数表示最大三个正向配对改善占全部正向改善的比例，与净平均改善及负向电池共同报告。完整逐电池差值保留在long_horizon_paired_cells.csv。

## 随访内与随访外原始输出

| domain | method | region | query_points | negative_predictions | above1p2 | nonfinite | minimum | maximum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HUST | free_rate | beyond_cell_followup | 238878 | 81518 | 0 | 0 | -2.887656450 | 0.971204817 |
| HUST | free_rate | observed_future | 138422 | 0 | 0 | 0 | 0.550911605 | 1.105026603 |
| HUST | legacy_flow | beyond_cell_followup | 238878 | 0 | 0 | 0 | 0.348420143 | 0.934068680 |
| HUST | legacy_flow | observed_future | 138422 | 0 | 0 | 0 | 0.676294208 | 1.105041623 |
| HUST | monotone_spline | beyond_cell_followup | 238878 | 0 | 0 | 0 | 0.452111661 | 0.931256592 |
| HUST | monotone_spline | observed_future | 138422 | 0 | 0 | 0 | 0.706585944 | 1.105004907 |
| HUST | physics_basis | beyond_cell_followup | 238878 | 0 | 0 | 0 | 0.276898205 | 0.952646673 |
| HUST | physics_basis | observed_future | 138422 | 0 | 0 | 0 | 0.608415186 | 1.105008483 |
| HUST | query_mlp | beyond_cell_followup | 238878 | 3838 | 0 | 0 | -0.090919018 | 0.956902802 |
| HUST | query_mlp | observed_future | 138422 | 0 | 0 | 0 | 0.620155394 | 1.105039239 |
| MATR | free_rate | beyond_cell_followup | 704630 | 286958 | 0 | 0 | -20.140321732 | 0.961701155 |
| MATR | free_rate | observed_future | 123431 | 0 | 0 | 0 | 0.416238904 | 0.996155083 |
| MATR | legacy_flow | beyond_cell_followup | 704630 | 0 | 0 | 0 | 0.185225248 | 0.971800208 |
| MATR | legacy_flow | observed_future | 123431 | 0 | 0 | 0 | 0.603394628 | 0.997081459 |
| MATR | monotone_spline | beyond_cell_followup | 704630 | 0 | 0 | 0 | 0.177299291 | 0.966288090 |
| MATR | monotone_spline | observed_future | 123431 | 0 | 0 | 0 | 0.470669329 | 0.996101916 |
| MATR | physics_basis | beyond_cell_followup | 704630 | 0 | 0 | 0 | 0.050551474 | 0.965320230 |
| MATR | physics_basis | observed_future | 123431 | 0 | 0 | 0 | 0.587650418 | 0.996097863 |
| MATR | query_mlp | beyond_cell_followup | 704630 | 221821 | 0 | 0 | -2.896925926 | 0.973317266 |
| MATR | query_mlp | observed_future | 123431 | 0 | 0 | 0 | 0.520829797 | 0.996166527 |

原始预测保留负值和超过1.2的输出。5000圈查询提供数值诊断，随访外部分缺少真实SOH准确性验证。单调样条与三分量基函数具有非负非增的数学结构，自由非负速率积分可能产生负SOH。

## MATR_b1c2 的真实OOF误差

| phase | method | fold | mae | rmse | n_obs |
| --- | --- | --- | --- | --- | --- |
| N1 | resmlp | cell_fold_3 | 0.025347127 | 0.028264413 | 2136 |
| N1 | voltage_cnn | cell_fold_3 | 0.059882384 | 0.082054669 | 2136 |
| N1 | temporal_tcn | cell_fold_3 | 0.036403747 | 0.040387223 | 2136 |
| N1 | cnn_tcn | cell_fold_3 | 0.014426907 | 0.016845322 | 2136 |
| N1 | cnn_transformer | cell_fold_3 | 0.020108422 | 0.024199510 | 2136 |
| N1 | nested | cell_fold_3 | 0.059882384 | 0.082054669 | 2136 |
| N2 | query_mlp | cell_fold_3 | 0.059882384 | 0.082054669 | 2136 |
| N2 | legacy_flow | cell_fold_3 | 0.045039775 | 0.055273942 | 2136 |
| N2 | free_rate | cell_fold_3 | 0.037137428 | 0.048523330 | 2136 |
| N2 | monotone_spline | cell_fold_3 | 0.082834342 | 0.097441170 | 2136 |
| N2 | physics_basis | cell_fold_3 | 0.087003771 | 0.115890153 | 2136 |
| N2 | nested | cell_fold_3 | 0.082834342 | 0.097441170 | 2136 |
| HISTORICAL | M1 | cell_fold_3 | 0.013239598 | 0.016838167 | 2136 |

该表使用原电池外折，历史M4/M5中未见协议测试的同名电池具有另一训练划分，两种scope分别解释。所有最大误差案例见worst_cases.csv。

## Stanford、EOL与协议支持

Stanford仍为N0_BLOCKED，已有181颗供应曲线缺少满足本轮真实原生循环1–100的认证张量。MATR的169颗具有已观察80%容量阈值事件，HUST77颗全部右删失。本轮交付EOL资格。HUST77个精确协议各只有一颗，当前cell-disjoint结果描述这些已释放组合的留出预测，正式跨域共享训练、工况族留域评价及组内重复方差仍缺证据。

## 工程修正和保留的失败

交接原7项单测通过。新增4项回归后11项通过，真实输入CUDA检查98项通过。自由速率固定网格、有效循环间隔、ResMLP残差块、训练集float64统计、原子恢复、测试评分闸门和逐电池汇总均在正式训练前修正。初次CUDA自适应池化确定性失败、正式启动前相对路径表示失败均保留。两类失败发生时正式优化器更新均为0。源头问题修复后固定8点平均池化及规范根路径重新通过检查。

## 优势、风险与下一步判断

优势为独立结构、完整原始队列、清晰信息预算、严格折内选模及可重放产物。风险包括各模型容量差异、M1额外物理特征和辅助损失、同一验证集上的多阶段选择、单种子、历史暴露和随访支持差异。验收与当前数值计数的置信度高，单种子对机制、跨域与未来新批次的证据置信度低。

N3的研究价值在于固定当前可复核结构后，单独检验显式全局/局部物理统计的增量，并控制与M1之间的信息预算和损失差异。当前N1/N2结果已提供确定性参照，概率支线的收益与校准尚需独立合同和评价。

## 拓展思考

编码器、解码器和物理信息预算分别具有可检验的作用，本轮提供了前两者在简洁VIQ/deltaQ输入下的对照。

## 你可能想知道

每外折可选择不同编码器和解码器，nested-OOF反映该验证选择程序的表现。代码没有用全局测试排名确定唯一模型。所有源码、权重、训练/验证曲线、逐电池预测、资格表、哈希和验收记录随功能分支交付。
