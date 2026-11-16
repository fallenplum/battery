# D3 未来标签函数表达诊断

本轮沿用种子42。MATR169、HUST77分别做原固定五折，原始K100与全部真实未来观测mask保持一致。训练使用150epoch、batch8、AdamW lr0.0003、wd0.01，选权为本折validation trajectory MAE首个严格最优。单位为SOH fraction。全部合法成绩按物理cell观测点先平均，再在cell间等权。当前队列存在历史研究暴露，结论属于内部回顾性研究。

所有配对区间均为当前单种子预测上的2000次条件bootstrap，无法估计训练种子变异。protocol区间对应协议macro estimand，cell区间对应cell等权estimand。HUST77协议各一颗电池，无法识别协议内方差。多重探索对比未进行家族误差率调整，不能据单一阳性比较升级论文结论。

所有本节函数拟合的prediction_scope均为ORACLE_DIAGNOSTIC。每颗cell允许读取其完整真实未来SOH，八初值float64 L-BFGS-B最小化绝对误差。每个family的全部初值、参数、状态及预测保存在独立诊断目录，未提供给合法训练器。

physics_shared的time_c与smooth由对应折train未来曲线联合学习，个体4参数在该cell全部future上拟合。physics_free另允许每cell自由time_c与smooth，共6参数。后者自由度更高。spline10沿用十段正rate指数累计，positive_rate8是八段正SOH损失rate积分，polynomial4是锚点固定四次非单调多项式，linear1为非负线性损失。

## OOF身份下的诊断汇总

| prediction_scope | domain | arm | supported_cells | observations | cell_mean_mae | cell_mean_rmse | protocol_macro_mae | bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ORACLE_DIAGNOSTIC | HUST | linear1 | 77 | 138422 | 0.021398 | 0.028740 | 0.021398 | 0.001720 |
| ORACLE_DIAGNOSTIC | HUST | physics_free | 77 | 138422 | 0.000553 | 0.000716 | 0.000553 | 0.000082 |
| ORACLE_DIAGNOSTIC | HUST | physics_shared | 77 | 138422 | 0.001005 | 0.001247 | 0.001005 | 0.000096 |
| ORACLE_DIAGNOSTIC | HUST | polynomial4 | 77 | 138422 | 0.000801 | 0.001071 | 0.000801 | -0.000087 |
| ORACLE_DIAGNOSTIC | HUST | positive_rate8 | 77 | 138422 | 0.000991 | 0.001374 | 0.000991 | 0.000142 |
| ORACLE_DIAGNOSTIC | HUST | spline10 | 77 | 138422 | 0.001899 | 0.002658 | 0.001899 | 0.000155 |
| ORACLE_DIAGNOSTIC | MATR | linear1 | 169 | 123431 | 0.024732 | 0.043730 | 0.022642 | 0.011540 |
| ORACLE_DIAGNOSTIC | MATR | physics_free | 169 | 123431 | 0.001283 | 0.001762 | 0.001131 | -0.000093 |
| ORACLE_DIAGNOSTIC | MATR | physics_shared | 169 | 123431 | 0.003006 | 0.004040 | 0.002538 | 0.000362 |
| ORACLE_DIAGNOSTIC | MATR | polynomial4 | 169 | 123431 | 0.002108 | 0.003120 | 0.001641 | -0.000049 |
| ORACLE_DIAGNOSTIC | MATR | positive_rate8 | 169 | 123431 | 0.009373 | 0.015688 | 0.011271 | 0.001852 |
| ORACLE_DIAGNOSTIC | MATR | spline10 | 169 | 123431 | 0.013823 | 0.023965 | 0.016313 | 0.004708 |

## 多初值优化状态

| prediction_scope | domain | family | starts | converged | median_iterations | median_gradient_inf |
| --- | --- | --- | --- | --- | --- | --- |
| ORACLE_DIAGNOSTIC | HUST | linear1 | 616 | 616 | 7.000000 | 0.000033 |
| ORACLE_DIAGNOSTIC | HUST | physics_free | 616 | 615 | 59.000000 | 0.000100 |
| ORACLE_DIAGNOSTIC | HUST | physics_shared | 3080 | 3080 | 35.000000 | 0.000067 |
| ORACLE_DIAGNOSTIC | HUST | polynomial4 | 616 | 616 | 43.000000 | 0.002514 |
| ORACLE_DIAGNOSTIC | HUST | positive_rate8 | 616 | 616 | 46.000000 | 0.000026 |
| ORACLE_DIAGNOSTIC | HUST | spline10 | 616 | 616 | 35.000000 | 0.000026 |
| ORACLE_DIAGNOSTIC | MATR | linear1 | 1352 | 1352 | 7.000000 | 0.000026 |
| ORACLE_DIAGNOSTIC | MATR | physics_free | 1352 | 1350 | 54.000000 | 0.000365 |
| ORACLE_DIAGNOSTIC | MATR | physics_shared | 6760 | 6760 | 8.000000 | 0.000020 |
| ORACLE_DIAGNOSTIC | MATR | polynomial4 | 1352 | 1352 | 43.000000 | 0.000838 |
| ORACLE_DIAGNOSTIC | MATR | positive_rate8 | 1352 | 1352 | 19.000000 | 0.000032 |
| ORACLE_DIAGNOSTIC | MATR | spline10 | 1352 | 1352 | 15.000000 | 0.000028 |

数值搜索不保证全局最小。L-BFGS-B对非光滑MAE可能提前停止或达到预算，所有失败状态仍保留。softplus极小值在浮点中可能接近正边界，Jacobian病态与共享尺度漂移需结合D4解释。训练、验证、测试身份分别列在oracle_summary_metrics.csv，完整四时距、EOL前后与SOH回升在oracle_cell_metrics.csv。fit_gap包含函数自由度、共享尺度学习、参数映射与数值优化差异，无法解释为纯信息下界。优势为同锚点与同观测支持下的有限自由度对照。风险为未来标签访问和局部优化，函数能拟合的结论置信度为中，电化学机制识别证据仍不足。
