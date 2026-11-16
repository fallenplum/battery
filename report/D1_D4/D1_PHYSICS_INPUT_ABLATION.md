# D1 显式早期物理特征消融

本轮沿用种子42。MATR169、HUST77分别做原固定五折，原始K100与全部真实未来观测mask保持一致。训练使用150epoch、batch8、AdamW lr0.0003、wd0.01，选权为本折validation trajectory MAE首个严格最优。单位为SOH fraction。全部合法成绩按物理cell观测点先平均，再在cell间等权。当前队列存在历史研究暴露，结论属于内部回顾性研究。

所有配对区间均为当前单种子预测上的2000次条件bootstrap，无法估计训练种子变异。protocol区间对应协议macro estimand，cell区间对应cell等权estimand。HUST77协议各一颗电池，无法识别协议内方差。多重探索对比未进行家族误差率调整，不能据单一阳性比较升级论文结论。

## 全轨迹

| prediction_scope | domain | arm | supported_cells | observations | cell_mean_mae | cell_mean_rmse | protocol_macro_mae | bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | HUST | D1-0 | 77 | 138422 | 0.023898 | 0.032261 | 0.023898 | 0.001084 |
| LEGAL_EARLY_FORECAST | HUST | D1-0-capmatch | 77 | 138422 | 0.023937 | 0.032331 | 0.023937 | -0.001356 |
| LEGAL_EARLY_FORECAST | HUST | D1-G | 77 | 138422 | 0.024209 | 0.032026 | 0.024209 | -0.000791 |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | 77 | 138422 | 0.021534 | 0.029714 | 0.021534 | 0.002299 |
| LEGAL_EARLY_FORECAST | HUST | D1-L | 77 | 138422 | 0.021348 | 0.029444 | 0.021348 | 0.003385 |
| LEGAL_EARLY_FORECAST | MATR | D1-0 | 169 | 123431 | 0.030660 | 0.051671 | 0.030227 | 0.014625 |
| LEGAL_EARLY_FORECAST | MATR | D1-0-capmatch | 169 | 123431 | 0.031762 | 0.052696 | 0.030985 | 0.013977 |
| LEGAL_EARLY_FORECAST | MATR | D1-G | 169 | 123431 | 0.024712 | 0.042445 | 0.023704 | 0.010195 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | 169 | 123431 | 0.023226 | 0.039397 | 0.021964 | 0.008913 |
| LEGAL_EARLY_FORECAST | MATR | D1-L | 169 | 123431 | 0.023096 | 0.038479 | 0.021255 | 0.008775 |
| LEGAL_EARLY_FORECAST | MATR | HISTORICAL-M1 | 169 | 123431 | 0.018762 | 0.031579 | 0.016989 | 0.004068 |

## 受控配对

| prediction_scope | domain | window | reference | candidate | n_cells | mean_improvement | median_improvement | better | worse | cell_ci_low | cell_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | MATR | all | D1-0 | D1-G | 169 | 0.005948 | 0.002971 | 130 | 39 | 0.003925 | 0.008179 |
| LEGAL_EARLY_FORECAST | MATR | all | D1-0 | D1-L | 169 | 0.007565 | 0.005911 | 137 | 32 | 0.005817 | 0.009558 |
| LEGAL_EARLY_FORECAST | MATR | all | D1-0 | D1-GL | 169 | 0.007435 | 0.004778 | 137 | 32 | 0.005553 | 0.009471 |
| LEGAL_EARLY_FORECAST | MATR | all | D1-0 | D1-0-capmatch | 169 | -0.001101 | -0.001022 | 64 | 105 | -0.002629 | 0.000385 |
| LEGAL_EARLY_FORECAST | MATR | all | D1-0-capmatch | D1-GL | 169 | 0.008536 | 0.006176 | 144 | 25 | 0.006372 | 0.010924 |
| LEGAL_EARLY_FORECAST | HUST | all | D1-0 | D1-G | 77 | -0.000310 | 0.000494 | 42 | 35 | -0.002883 | 0.002132 |
| LEGAL_EARLY_FORECAST | HUST | all | D1-0 | D1-L | 77 | 0.002550 | -0.000022 | 38 | 39 | 0.000349 | 0.004779 |
| LEGAL_EARLY_FORECAST | HUST | all | D1-0 | D1-GL | 77 | 0.002364 | 0.000063 | 40 | 37 | 0.000122 | 0.004794 |
| LEGAL_EARLY_FORECAST | HUST | all | D1-0 | D1-0-capmatch | 77 | -0.000039 | -0.000406 | 36 | 41 | -0.001835 | 0.001904 |
| LEGAL_EARLY_FORECAST | HUST | all | D1-0-capmatch | D1-GL | 77 | 0.002403 | 0.000759 | 42 | 35 | 0.000277 | 0.004562 |

## 容量与时距

| prediction_scope | domain | arm | parameters | extra_params |
| --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | HUST | D1-0-capmatch | 78813.800000 | 22896.000000 |
| LEGAL_EARLY_FORECAST | HUST | D1-G | 62573.800000 | 6656.000000 |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | 78701.800000 | 22784.000000 |
| LEGAL_EARLY_FORECAST | HUST | D1-L | 72237.800000 | 16320.000000 |
| LEGAL_EARLY_FORECAST | HUST | D2-F0L1 | 55917.800000 | 0.000000 |
| LEGAL_EARLY_FORECAST | HUST | D2-F1L1 | 78701.800000 | 22784.000000 |
| LEGAL_EARLY_FORECAST | HUST | D2-GL-rate-physics | 76786.800000 | 22784.000000 |
| LEGAL_EARLY_FORECAST | HUST | D2-GL-rate-spline | 77174.800000 | 22784.000000 |
| LEGAL_EARLY_FORECAST | MATR | D1-0-capmatch | 92708.200000 | 22896.000000 |
| LEGAL_EARLY_FORECAST | MATR | D1-G | 76468.200000 | 6656.000000 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | 92596.200000 | 22784.000000 |
| LEGAL_EARLY_FORECAST | MATR | D1-L | 86132.200000 | 16320.000000 |
| LEGAL_EARLY_FORECAST | MATR | D2-F0L1 | 69812.200000 | 0.000000 |
| LEGAL_EARLY_FORECAST | MATR | D2-F1L1 | 92596.200000 | 22784.000000 |
| LEGAL_EARLY_FORECAST | MATR | D2-GL-rate-physics | 90681.200000 | 22784.000000 |
| LEGAL_EARLY_FORECAST | MATR | D2-GL-rate-spline | 91069.200000 | 22784.000000 |

| prediction_scope | domain | arm | window | supported_cells | cell_mean_mae | bias |
| --- | --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | HUST | D1-0 | 1001_plus | 77 | 0.041783 | 0.012187 |
| LEGAL_EARLY_FORECAST | HUST | D1-0 | 101_200 | 77 | 0.001731 | 0.001172 |
| LEGAL_EARLY_FORECAST | HUST | D1-0 | 201_500 | 77 | 0.005649 | 0.000015 |
| LEGAL_EARLY_FORECAST | HUST | D1-0 | 501_1000 | 77 | 0.013323 | -0.004277 |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | 1001_plus | 77 | 0.038067 | 0.015332 |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | 101_200 | 77 | 0.001537 | 0.001006 |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | 201_500 | 77 | 0.005114 | -0.000691 |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | 501_1000 | 77 | 0.012972 | -0.005161 |
| LEGAL_EARLY_FORECAST | MATR | D1-0 | 1001_plus | 40 | 0.095334 | 0.043339 |
| LEGAL_EARLY_FORECAST | MATR | D1-0 | 101_200 | 169 | 0.002470 | -0.000292 |
| LEGAL_EARLY_FORECAST | MATR | D1-0 | 201_500 | 168 | 0.015651 | 0.004872 |
| LEGAL_EARLY_FORECAST | MATR | D1-0 | 501_1000 | 151 | 0.071040 | 0.053832 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | 1001_plus | 40 | 0.079956 | 0.049368 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | 101_200 | 169 | 0.002850 | -0.000225 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | 201_500 | 168 | 0.011541 | 0.000431 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | 501_1000 | 151 | 0.052788 | 0.040725 |

优势为同一原始支路、同一UID与相近参数容量对照。风险为额外分支改变优化路径，容量近似匹配仍无法消除函数形式差异。完整reference VIQ可由现有四通道近似重建，具体舍入误差与旧架构的表征差异见UNMATCHED_INFORMATION.md。额外分支使用固定派生初始化种子420014，原主干与decoder初值沿用seed42。逐cell反例与长寿命贡献保留在worst_cases.csv和long_horizon_concentration.csv。输入增量的方向只适用于当前折与单种子，置信度为中。
