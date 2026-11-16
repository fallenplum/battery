# D4 参数稳定性、速率可预测性与实际梯度

本轮沿用种子42。MATR169、HUST77分别做原固定五折，原始K100与全部真实未来观测mask保持一致。训练使用150epoch、batch8、AdamW lr0.0003、wd0.01，选权为本折validation trajectory MAE首个严格最优。单位为SOH fraction。全部合法成绩按物理cell观测点先平均，再在cell间等权。当前队列存在历史研究暴露，结论属于内部回顾性研究。

所有配对区间均为当前单种子预测上的2000次条件bootstrap，无法估计训练种子变异。protocol区间对应协议macro estimand，cell区间对应cell等权estimand。HUST77协议各一颗电池，无法识别协议内方差。多重探索对比未进行家族误差率调整，不能据单一阳性比较升级论文结论。

## 多初值稳定性

近等价集合同时满足MAE≤best×1.05且差≤0.002，阈值敏感性1%与10%预先登记。

| prediction_scope | domain | family | parameter | cells | median_range | max_range | median_equivalent_starts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ORACLE_DIAGNOSTIC | HUST | physics_free | a | 77 | 20.156573 | 20392.067882 | 7.000000 |
| ORACLE_DIAGNOSTIC | HUST | physics_free | acceleration_c | 77 | 0.003803 | 1.637351 | 7.000000 |
| ORACLE_DIAGNOSTIC | HUST | physics_free | b | 77 | 0.043558 | 0.116208 | 7.000000 |
| ORACLE_DIAGNOSTIC | HUST | physics_free | smooth | 77 | 0.002385 | 0.027636 | 7.000000 |
| ORACLE_DIAGNOSTIC | HUST | physics_free | tau | 77 | 0.001300 | 0.451788 | 7.000000 |
| ORACLE_DIAGNOSTIC | HUST | physics_free | time_c | 77 | 694.831342 | 641390.582597 | 7.000000 |
| ORACLE_DIAGNOSTIC | HUST | physics_shared | a | 77 | 0.036600 | 1.098294 | 7.000000 |
| ORACLE_DIAGNOSTIC | HUST | physics_shared | acceleration_c | 77 | 0.000287 | 0.012527 | 7.000000 |
| ORACLE_DIAGNOSTIC | HUST | physics_shared | b | 77 | 0.003098 | 0.097925 | 7.000000 |
| ORACLE_DIAGNOSTIC | HUST | physics_shared | smooth | 77 | 0.000000 | 0.000000 | 7.000000 |
| ORACLE_DIAGNOSTIC | HUST | physics_shared | tau | 77 | 0.000687 | 0.017603 | 7.000000 |
| ORACLE_DIAGNOSTIC | HUST | physics_shared | time_c | 77 | 0.000000 | 0.000000 | 7.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_free | a | 169 | 0.002091 | 73637.026718 | 2.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_free | acceleration_c | 169 | 0.000852 | 388.156299 | 2.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_free | b | 169 | 2.29477e-06 | 0.445818 | 2.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_free | smooth | 169 | 5.20764e-15 | 0.035324 | 2.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_free | tau | 169 | 0.000074 | 1.184848 | 2.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_free | time_c | 169 | 10.794168 | 1.27729e+06 | 2.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_shared | a | 169 | 0.033345 | 2.982966 | 2.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_shared | acceleration_c | 169 | 0.000034 | 0.060028 | 2.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_shared | b | 169 | 0.000556 | 0.063989 | 2.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_shared | smooth | 169 | 0.000000 | 0.000000 | 2.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_shared | tau | 169 | 9.56568e-06 | 0.053084 | 2.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_shared | time_c | 169 | 0.000000 | 0.000000 | 2.000000 |

## 缩放Jacobian

| prediction_scope | domain | family | followup_group | cells | median_condition | median_near_zero |
| --- | --- | --- | --- | --- | --- | --- |
| ORACLE_DIAGNOSTIC | HUST | physics_free | over_1000 | 77 | 2.82883e+07 | 1.000000 |
| ORACLE_DIAGNOSTIC | HUST | physics_shared | over_1000 | 77 | 15994.016568 | 0.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_free | 1000_or_less | 129 | 7.34814e+16 | 2.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_free | over_1000 | 40 | 2.49038e+14 | 1.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_shared | 1000_or_less | 129 | 3.42474e+06 | 1.000000 |
| ORACLE_DIAGNOSTIC | MATR | physics_shared | over_1000 | 40 | 462333.606617 | 0.000000 |

Jacobian用实际可观测query，以物理参数预定尺度[0.05,0.05,0.06,1,0.5,0.25]计算SVD。它只刻画当前位置与支持区间的局部病态。shared模式的两个共享参数固定，free模式有6个个体方向。profile子集固定为每域fold0排序最前3个train与3个test UID，未按拟合效果筛选。有效rate网格单位为SOH fraction/1000 cycles，SLSQP未收敛或约束残差较大的点保留在profiles.csv。

## 合法K100预测真实端点rate

| prediction_scope | domain | feature_set | window | n_cells | mae | rmse | r2 | increment_over_anchor_policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | MATR | anchor_policy | 200_500 | 151 | 0.000145 | 0.000192 | 0.220372 | 0.000000 |
| LEGAL_EARLY_FORECAST | MATR | anchor_policy | 500_1000 | 42 | 0.000142 | 0.000203 | 0.189884 | 0.000000 |
| LEGAL_EARLY_FORECAST | MATR | frozen_z_D1-GL | 200_500 | 151 | 0.000087 | 0.000135 | 0.610708 | 0.000058 |
| LEGAL_EARLY_FORECAST | MATR | frozen_z_D1-GL | 500_1000 | 42 | 0.000138 | 0.000203 | 0.187752 | 3.51608e-06 |
| LEGAL_EARLY_FORECAST | MATR | frozen_z_D2-F1L1 | 200_500 | 151 | 0.000083 | 0.000127 | 0.656930 | 0.000063 |
| LEGAL_EARLY_FORECAST | MATR | frozen_z_D2-F1L1 | 500_1000 | 42 | 0.000145 | 0.000210 | 0.130318 | -3.31505e-06 |
| LEGAL_EARLY_FORECAST | MATR | frozen_z_D2-GL-rate-physics | 200_500 | 151 | 0.000090 | 0.000135 | 0.614886 | 0.000055 |
| LEGAL_EARLY_FORECAST | MATR | frozen_z_D2-GL-rate-physics | 500_1000 | 42 | 0.000147 | 0.000205 | 0.172325 | -5.39864e-06 |
| LEGAL_EARLY_FORECAST | MATR | plus_G | 200_500 | 151 | 0.000081 | 0.000127 | 0.657808 | 0.000064 |
| LEGAL_EARLY_FORECAST | MATR | plus_G | 500_1000 | 42 | 0.000152 | 0.000213 | 0.108453 | -0.000010 |
| LEGAL_EARLY_FORECAST | MATR | plus_GL | 200_500 | 151 | 0.000081 | 0.000129 | 0.645148 | 0.000064 |
| LEGAL_EARLY_FORECAST | MATR | plus_GL | 500_1000 | 42 | 0.000150 | 0.000217 | 0.075056 | -8.70809e-06 |
| LEGAL_EARLY_FORECAST | MATR | plus_L | 200_500 | 151 | 0.000081 | 0.000130 | 0.640757 | 0.000064 |
| LEGAL_EARLY_FORECAST | MATR | plus_L | 500_1000 | 42 | 0.000152 | 0.000218 | 0.061650 | -0.000010 |
| LEGAL_EARLY_FORECAST | MATR | raw_statistics | 200_500 | 151 | 0.000095 | 0.000131 | 0.633527 | 0.000050 |
| LEGAL_EARLY_FORECAST | MATR | raw_statistics | 500_1000 | 42 | 0.000150 | 0.000209 | 0.136490 | -8.67316e-06 |
| LEGAL_EARLY_FORECAST | HUST | anchor_policy | 200_500 | 77 | 0.000010 | 0.000015 | -0.056055 | 0.000000 |
| LEGAL_EARLY_FORECAST | HUST | anchor_policy | 500_1000 | 77 | 0.000031 | 0.000043 | 0.209193 | 0.000000 |
| LEGAL_EARLY_FORECAST | HUST | frozen_z_D1-GL | 200_500 | 77 | 9.56245e-06 | 0.000014 | 0.046924 | 6.70317e-07 |
| LEGAL_EARLY_FORECAST | HUST | frozen_z_D1-GL | 500_1000 | 77 | 0.000030 | 0.000042 | 0.232613 | 5.15146e-07 |
| LEGAL_EARLY_FORECAST | HUST | frozen_z_D2-F1L1 | 200_500 | 77 | 9.47684e-06 | 0.000014 | 0.077683 | 7.55933e-07 |
| LEGAL_EARLY_FORECAST | HUST | frozen_z_D2-F1L1 | 500_1000 | 77 | 0.000031 | 0.000041 | 0.257749 | 2.38933e-07 |
| LEGAL_EARLY_FORECAST | HUST | frozen_z_D2-GL-rate-physics | 200_500 | 77 | 9.37967e-06 | 0.000014 | 0.138539 | 8.53098e-07 |
| LEGAL_EARLY_FORECAST | HUST | frozen_z_D2-GL-rate-physics | 500_1000 | 77 | 0.000030 | 0.000041 | 0.255638 | 5.55557e-07 |
| LEGAL_EARLY_FORECAST | HUST | plus_G | 200_500 | 77 | 9.02212e-06 | 0.000014 | 0.111780 | 1.21065e-06 |
| LEGAL_EARLY_FORECAST | HUST | plus_G | 500_1000 | 77 | 0.000030 | 0.000039 | 0.335192 | 1.0097e-06 |
| LEGAL_EARLY_FORECAST | HUST | plus_GL | 200_500 | 77 | 7.68663e-06 | 0.000011 | 0.420526 | 2.54613e-06 |
| LEGAL_EARLY_FORECAST | HUST | plus_GL | 500_1000 | 77 | 0.000030 | 0.000040 | 0.301011 | 7.59288e-07 |
| LEGAL_EARLY_FORECAST | HUST | plus_L | 200_500 | 77 | 7.85448e-06 | 0.000011 | 0.387518 | 2.37829e-06 |
| LEGAL_EARLY_FORECAST | HUST | plus_L | 500_1000 | 77 | 0.000029 | 0.000039 | 0.343646 | 1.6981e-06 |
| LEGAL_EARLY_FORECAST | HUST | raw_statistics | 200_500 | 77 | 9.56498e-06 | 0.000014 | 0.142954 | 6.67791e-07 |
| LEGAL_EARLY_FORECAST | HUST | raw_statistics | 500_1000 | 77 | 0.000024 | 0.000032 | 0.561498 | 7.44404e-06 |

轻头只拟合本折有支持的train电池，StandardScaler也只fit该train支持集，alpha仅由本折validation选择，未合并trainval重拟合。每窗口按原UID的OOF test评分。冻结z来自相同折已冻结D1/D2网络，属于已有监督表征的条件诊断。共160个选定rate proxy模型，6候选alpha对应960次Ridge拟合，独立记录于RATE_PROXY_MODELS.json。MATR第二段n=42，R²对少量长寿命电池敏感。

## 实际physics_basis训练probe

| prediction_scope | domain | fold | epoch | time_c | smooth | raw_tau_grad_norm | raw_a_grad_norm | raw_b_grad_norm | raw_c_grad_norm | tau_within_followup_fraction | median_tau_cycle | acceleration_fraction_501_1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_0 | 1 | 0.421217 | 0.231884 | 0.000342 | 0.002407 | 0.002987 | 0.000090 | 0.250000 | 2089.281738 | 0.000965 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_0 | 12 | 0.422897 | 0.225587 | 0.008996 | 0.000275 | 0.000944 | 0.002063 | 1.000000 | 660.889008 | 0.639598 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_0 | 150 | 0.428854 | 0.170109 | 0.020036 | 0.000582 | 0.000621 | 0.000273 | 0.250000 | 931.461761 | 0.596254 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_1 | 1 | 0.420912 | 0.231479 | 0.000425 | 0.002924 | 0.002389 | 0.000103 | 0.250000 | 2146.655396 | 0.000609 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_1 | 12 | 0.421756 | 0.227204 | 0.008482 | 0.000315 | 0.001316 | 0.001899 | 0.750000 | 789.103149 | 0.519302 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_1 | 150 | 0.397908 | 0.171944 | 0.022490 | 8.17338e-06 | 0.002014 | 0.000253 | 0.250000 | 938.593872 | 0.567950 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_2 | 1 | 0.420954 | 0.231799 | 0.000181 | 0.002887 | 0.002487 | 0.000053 | 0.125000 | 2052.640991 | 0.000829 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_2 | 12 | 0.421488 | 0.228190 | 0.007347 | 0.000488 | 0.001777 | 0.001131 | 0.750000 | 873.582886 | 0.514509 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_2 | 150 | 0.403538 | 0.175255 | 0.027609 | 5.58864e-06 | 0.000693 | 0.000326 | 0.125000 | 976.777191 | 0.739461 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_3 | 1 | 0.420982 | 0.232110 | 0.000218 | 0.002350 | 0.001835 | 0.000039 | 0.000000 | 2046.231140 | 0.000844 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_3 | 12 | 0.422407 | 0.228076 | 0.007678 | 0.000374 | 0.001578 | 0.001801 | 1.000000 | 708.253387 | 0.650020 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_3 | 150 | 0.403610 | 0.174526 | 0.023832 | 0.000018 | 0.001029 | 0.000305 | 0.125000 | 903.230621 | 0.758912 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_4 | 1 | 0.421132 | 0.231554 | 0.000128 | 0.003338 | 0.002727 | 0.000026 | 0.000000 | 2296.664185 | 0.000328 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_4 | 12 | 0.423936 | 0.232540 | 0.005858 | 0.000449 | 0.001441 | 0.002004 | 0.875000 | 721.972504 | 0.573773 |
| LEGAL_EARLY_FORECAST | MATR | cell_fold_4 | 150 | 0.411531 | 0.176899 | 0.028230 | 0.000604 | 0.001082 | 0.000871 | 0.250000 | 991.621063 | 0.632585 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_0 | 1 | 0.421394 | 0.231648 | 0.002291 | 0.003907 | 0.003502 | 0.000848 | 1.000000 | 1010.852936 | 0.043319 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_0 | 12 | 0.423117 | 0.229502 | 0.010108 | 0.002377 | 0.006714 | 0.003392 | 1.000000 | 931.933014 | 0.213530 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_0 | 150 | 0.425733 | 0.216721 | 0.012720 | 0.000724 | 0.004629 | 0.001100 | 1.000000 | 1170.339417 | 0.253233 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_1 | 1 | 0.421661 | 0.231644 | 0.004296 | 0.002520 | 0.006404 | 0.002299 | 1.000000 | 972.888367 | 0.075970 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_1 | 12 | 0.423075 | 0.229579 | 0.010380 | 0.001796 | 0.005592 | 0.003877 | 1.000000 | 966.126160 | 0.218469 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_1 | 150 | 0.427318 | 0.219415 | 0.017653 | 0.001842 | 0.006079 | 0.002365 | 1.000000 | 1227.174500 | 0.254691 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_2 | 1 | 0.421191 | 0.231726 | 0.004346 | 0.005425 | 0.007064 | 0.001750 | 1.000000 | 1202.748596 | 0.032601 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_2 | 12 | 0.422681 | 0.229394 | 0.009721 | 0.002089 | 0.005177 | 0.002706 | 1.000000 | 952.014313 | 0.210224 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_2 | 150 | 0.424753 | 0.215232 | 0.019393 | 0.001107 | 0.006481 | 0.002384 | 1.000000 | 1514.200439 | 0.187449 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_3 | 1 | 0.421694 | 0.231611 | 0.002921 | 0.004278 | 0.004336 | 0.001091 | 1.000000 | 1068.726562 | 0.037092 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_3 | 12 | 0.423501 | 0.229832 | 0.008980 | 0.002111 | 0.005264 | 0.002838 | 1.000000 | 905.875092 | 0.239375 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_3 | 150 | 0.428356 | 0.217272 | 0.013927 | 0.000575 | 0.004000 | 0.001329 | 1.000000 | 1420.074036 | 0.180010 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_4 | 1 | 0.421666 | 0.231659 | 0.002300 | 0.002449 | 0.004436 | 0.000862 | 0.875000 | 1292.408813 | 0.020691 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_4 | 12 | 0.422968 | 0.229752 | 0.009547 | 0.002224 | 0.006352 | 0.004907 | 1.000000 | 855.589508 | 0.277439 |
| LEGAL_EARLY_FORECAST | HUST | cell_fold_4 | 150 | 0.425988 | 0.218143 | 0.019094 | 0.001922 | 0.005067 | 0.002121 | 1.000000 | 1122.555969 | 0.269046 |

每epoch固定本折前8个train UID进行无更新probe，记录raw与a/b/c/tau、共享尺度、分布、梯度与各时距加速贡献。probe前后恢复随机态与train/eval模式。完整150epoch可追溯。此次使用原始初始化，未运行可选tau初值控制。因此梯度弱只支持当前优化路径解释，无法确立初始化的因果作用。HUST全部右删失，无真实EOL预测评价。优势为拟合稳定性、真实rate代理和实际网络梯度分别可核查。风险为有限多初值、局部SVD与短随访，参数唯一性与真实电化学机制的置信度为低。
