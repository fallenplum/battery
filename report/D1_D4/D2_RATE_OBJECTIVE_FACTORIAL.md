# D2 速率辅助目标正交对照

本轮沿用种子42。MATR169、HUST77分别做原固定五折，原始K100与全部真实未来观测mask保持一致。训练使用150epoch、batch8、AdamW lr0.0003、wd0.01，选权为本折validation trajectory MAE首个严格最优。单位为SOH fraction。全部合法成绩按物理cell观测点先平均，再在cell间等权。当前队列存在历史研究暴露，结论属于内部回顾性研究。

所有配对区间均为当前单种子预测上的2000次条件bootstrap，无法估计训练种子变异。protocol区间对应协议macro estimand，cell区间对应cell等权estimand。HUST77协议各一颗电池，无法识别协议内方差。多重探索对比未进行家族误差率调整，不能据单一阳性比较升级论文结论。

F0L0为D1-0，F1L0为D1-GL，F0L1为D2-F0L1，F1L1为D2-F1L1。rate loss为supported-cell平均的supported-window MAE×1000×0.1。MATR两个窗口支持151与42颗，HUST两个窗口均支持77颗，mask保持全部主队列。历史trainer源码使用相同逐cell再逐window规则，目标并未改成全局window均值。

## 主表

| prediction_scope | domain | arm | supported_cells | observations | cell_mean_mae | cell_mean_rmse | protocol_macro_mae | bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | HUST | D1-0 | 77 | 138422 | 0.023898 | 0.032261 | 0.023898 | 0.001084 |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | 77 | 138422 | 0.021534 | 0.029714 | 0.021534 | 0.002299 |
| LEGAL_EARLY_FORECAST | HUST | D2-F0L1 | 77 | 138422 | 0.024775 | 0.033048 | 0.024775 | 0.000401 |
| LEGAL_EARLY_FORECAST | HUST | D2-F1L1 | 77 | 138422 | 0.022518 | 0.030451 | 0.022518 | -0.001070 |
| LEGAL_EARLY_FORECAST | HUST | D2-GL-rate-physics | 77 | 138422 | 0.020573 | 0.028835 | 0.020573 | 0.003194 |
| LEGAL_EARLY_FORECAST | HUST | D2-GL-rate-spline | 77 | 138422 | 0.018757 | 0.027013 | 0.018757 | 0.002097 |
| LEGAL_EARLY_FORECAST | MATR | D1-0 | 169 | 123431 | 0.030660 | 0.051671 | 0.030227 | 0.014625 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | 169 | 123431 | 0.023226 | 0.039397 | 0.021964 | 0.008913 |
| LEGAL_EARLY_FORECAST | MATR | D2-F0L1 | 169 | 123431 | 0.032485 | 0.053230 | 0.032825 | 0.010711 |
| LEGAL_EARLY_FORECAST | MATR | D2-F1L1 | 169 | 123431 | 0.025062 | 0.039273 | 0.024065 | 0.002893 |
| LEGAL_EARLY_FORECAST | MATR | D2-GL-rate-physics | 169 | 123431 | 0.024325 | 0.041540 | 0.023936 | 0.009311 |
| LEGAL_EARLY_FORECAST | MATR | D2-GL-rate-spline | 169 | 123431 | 0.024194 | 0.039280 | 0.024581 | 0.005251 |

## 正交效应

| prediction_scope | domain | window | effect | n_cells | mean_improvement | median_improvement | better | worse | cell_ci_low | cell_ci_high | protocol_macro_improvement | protocol_ci_low | protocol_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | MATR | all | feature_main | 169 | 0.007429 | 0.005224 | 138 | 31 | 0.005603 | 0.009382 | 0.008511 | 0.006008 | 0.011416 |
| LEGAL_EARLY_FORECAST | MATR | all | rate_main | 169 | -0.001830 | -0.000653 | 76 | 93 | -0.003074 | -0.000722 | -0.002350 | -0.004419 | -0.000674 |
| LEGAL_EARLY_FORECAST | MATR | all | interaction_gain | 169 | -0.000011 | -0.000920 | 78 | 91 | -0.001810 | 0.001790 | 0.000497 | -0.002158 | 0.003471 |
| LEGAL_EARLY_FORECAST | HUST | all | feature_main | 77 | 0.002311 | 0.000540 | 41 | 36 | 0.000088 | 0.004595 | 0.002311 | 0.000195 | 0.004609 |
| LEGAL_EARLY_FORECAST | HUST | all | rate_main | 77 | -0.000930 | 0.000438 | 45 | 32 | -0.002114 | 0.000116 | -0.000930 | -0.002008 | 0.000072 |
| LEGAL_EARLY_FORECAST | HUST | all | interaction_gain | 77 | -0.000107 | -0.000149 | 36 | 41 | -0.002427 | 0.002148 | -0.000107 | -0.002445 | 0.002182 |

interaction_gain=(F1L0−F1L1)−(F0L0−F0L1)，正值表示特征存在时rate辅助的收益更强。feature_main与rate_main取另一个因素两个水平上的平均改善。feature因素同时增加支路参数，需结合D1-capmatch解释。

## 501–1000圈偏差

| prediction_scope | domain | arm | supported_cells | cell_mean_mae | bias |
| --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | HUST | D1-0 | 77 | 0.013323 | -0.004277 |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | 77 | 0.012972 | -0.005161 |
| LEGAL_EARLY_FORECAST | HUST | D2-F0L1 | 77 | 0.013358 | -0.003432 |
| LEGAL_EARLY_FORECAST | HUST | D2-F1L1 | 77 | 0.013427 | -0.006074 |
| LEGAL_EARLY_FORECAST | HUST | D2-GL-rate-physics | 77 | 0.008659 | 0.002010 |
| LEGAL_EARLY_FORECAST | HUST | N2-physics_basis | 77 | 0.011096 | -0.000259 |
| LEGAL_EARLY_FORECAST | MATR | D1-0 | 151 | 0.071040 | 0.053832 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | 151 | 0.052788 | 0.040725 |
| LEGAL_EARLY_FORECAST | MATR | D2-F0L1 | 151 | 0.068874 | 0.047937 |
| LEGAL_EARLY_FORECAST | MATR | D2-F1L1 | 151 | 0.046018 | 0.029798 |
| LEGAL_EARLY_FORECAST | MATR | D2-GL-rate-physics | 151 | 0.053452 | 0.043634 |
| LEGAL_EARLY_FORECAST | MATR | N2-physics_basis | 151 | 0.071079 | 0.057277 |

## 损失与梯度

| prediction_scope | domain | arm | mean_main_loss | mean_weighted_rate | mean_gradient_cosine | conflicting_epochs |
| --- | --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | HUST | D1-0-capmatch | 0.021216 | 0.000000 | 0.208823 | 61.200000 |
| LEGAL_EARLY_FORECAST | HUST | D1-G | 0.020025 | 0.000000 | 0.202382 | 62.400000 |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | 0.020195 | 0.000000 | 0.189385 | 63.400000 |
| LEGAL_EARLY_FORECAST | HUST | D1-L | 0.020447 | 0.000000 | 0.240669 | 57.600000 |
| LEGAL_EARLY_FORECAST | HUST | D2-F0L1 | 0.020240 | 0.002457 | 0.122954 | 67.200000 |
| LEGAL_EARLY_FORECAST | HUST | D2-F1L1 | 0.020311 | 0.002475 | 0.073446 | 71.200000 |
| LEGAL_EARLY_FORECAST | HUST | D2-GL-rate-physics | 0.009429 | 0.001098 | 0.479823 | 36.400000 |
| LEGAL_EARLY_FORECAST | HUST | D2-GL-rate-spline | 0.007213 | 0.000710 | 0.427889 | 17.000000 |
| LEGAL_EARLY_FORECAST | MATR | D1-0-capmatch | 0.023750 | 0.000000 | 0.358541 | 42.200000 |
| LEGAL_EARLY_FORECAST | MATR | D1-G | 0.022724 | 0.000000 | 0.297436 | 50.200000 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | 0.021986 | 0.000000 | 0.292860 | 52.200000 |
| LEGAL_EARLY_FORECAST | MATR | D1-L | 0.022067 | 0.000000 | 0.313605 | 51.200000 |
| LEGAL_EARLY_FORECAST | MATR | D2-F0L1 | 0.024293 | 0.008774 | 0.175585 | 60.600000 |
| LEGAL_EARLY_FORECAST | MATR | D2-F1L1 | 0.024226 | 0.008525 | 0.176882 | 60.200000 |
| LEGAL_EARLY_FORECAST | MATR | D2-GL-rate-physics | 0.018718 | 0.006267 | 0.206764 | 59.800000 |
| LEGAL_EARLY_FORECAST | MATR | D2-GL-rate-spline | 0.019483 | 0.007180 | -0.250303 | 101.600000 |

优势为固定lambda与正交复用，未按测试分数选择目标权重。风险为辅助标签支持集中在有长随访的电池，MATR第二段仅42颗，训练信号分布存在选择性。各epoch实际支持数及固定训练probe的梯度夹角完整保留，置信度为中。
