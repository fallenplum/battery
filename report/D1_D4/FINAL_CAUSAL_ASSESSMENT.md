# D1–D4 失效归因裁决

本轮沿用种子42。MATR169、HUST77分别做原固定五折，原始K100与全部真实未来观测mask保持一致。训练使用150epoch、batch8、AdamW lr0.0003、wd0.01，选权为本折validation trajectory MAE首个严格最优。单位为SOH fraction。全部合法成绩按物理cell观测点先平均，再在cell间等权。当前队列存在历史研究暴露，结论属于内部回顾性研究。

所有配对区间均为当前单种子预测上的2000次条件bootstrap，无法估计训练种子变异。protocol区间对应协议macro estimand，cell区间对应cell等权estimand。HUST77协议各一颗电池，无法识别协议内方差。多重探索对比未进行家族误差率调整，不能据单一阳性比较升级论文结论。

## 合法完整轨迹结果

| prediction_scope | domain | arm | supported_cells | observations | cell_mean_mae | cell_mean_rmse | protocol_macro_mae | bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | HUST | D1-0 | 77 | 138422 | 0.023898 | 0.032261 | 0.023898 | 0.001084 |
| LEGAL_EARLY_FORECAST | HUST | D1-0-capmatch | 77 | 138422 | 0.023937 | 0.032331 | 0.023937 | -0.001356 |
| LEGAL_EARLY_FORECAST | HUST | D1-G | 77 | 138422 | 0.024209 | 0.032026 | 0.024209 | -0.000791 |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | 77 | 138422 | 0.021534 | 0.029714 | 0.021534 | 0.002299 |
| LEGAL_EARLY_FORECAST | HUST | D1-L | 77 | 138422 | 0.021348 | 0.029444 | 0.021348 | 0.003385 |
| LEGAL_EARLY_FORECAST | HUST | D2-F0L1 | 77 | 138422 | 0.024775 | 0.033048 | 0.024775 | 0.000401 |
| LEGAL_EARLY_FORECAST | HUST | D2-F1L1 | 77 | 138422 | 0.022518 | 0.030451 | 0.022518 | -0.001070 |
| LEGAL_EARLY_FORECAST | HUST | D2-GL-rate-physics | 77 | 138422 | 0.020573 | 0.028835 | 0.020573 | 0.003194 |
| LEGAL_EARLY_FORECAST | HUST | D2-GL-rate-spline | 77 | 138422 | 0.018757 | 0.027013 | 0.018757 | 0.002097 |
| LEGAL_EARLY_FORECAST | HUST | N2-free_rate | 77 | 138422 | 0.022358 | 0.030762 | 0.022358 | 0.002455 |
| LEGAL_EARLY_FORECAST | HUST | N2-monotone_spline | 77 | 138422 | 0.020500 | 0.028370 | 0.020500 | 0.001018 |
| LEGAL_EARLY_FORECAST | HUST | N2-physics_basis | 77 | 138422 | 0.021837 | 0.029912 | 0.021837 | 0.002485 |
| LEGAL_EARLY_FORECAST | MATR | D1-0 | 169 | 123431 | 0.030660 | 0.051671 | 0.030227 | 0.014625 |
| LEGAL_EARLY_FORECAST | MATR | D1-0-capmatch | 169 | 123431 | 0.031762 | 0.052696 | 0.030985 | 0.013977 |
| LEGAL_EARLY_FORECAST | MATR | D1-G | 169 | 123431 | 0.024712 | 0.042445 | 0.023704 | 0.010195 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | 169 | 123431 | 0.023226 | 0.039397 | 0.021964 | 0.008913 |
| LEGAL_EARLY_FORECAST | MATR | D1-L | 169 | 123431 | 0.023096 | 0.038479 | 0.021255 | 0.008775 |
| LEGAL_EARLY_FORECAST | MATR | D2-F0L1 | 169 | 123431 | 0.032485 | 0.053230 | 0.032825 | 0.010711 |
| LEGAL_EARLY_FORECAST | MATR | D2-F1L1 | 169 | 123431 | 0.025062 | 0.039273 | 0.024065 | 0.002893 |
| LEGAL_EARLY_FORECAST | MATR | D2-GL-rate-physics | 169 | 123431 | 0.024325 | 0.041540 | 0.023936 | 0.009311 |
| LEGAL_EARLY_FORECAST | MATR | D2-GL-rate-spline | 169 | 123431 | 0.024194 | 0.039280 | 0.024581 | 0.005251 |
| LEGAL_EARLY_FORECAST | MATR | HISTORICAL-M1 | 169 | 123431 | 0.018762 | 0.031579 | 0.016989 | 0.004068 |
| LEGAL_EARLY_FORECAST | MATR | N2-free_rate | 169 | 123431 | 0.030538 | 0.051591 | 0.029462 | 0.013408 |
| LEGAL_EARLY_FORECAST | MATR | N2-monotone_spline | 169 | 123431 | 0.030075 | 0.049478 | 0.030146 | 0.011036 |
| LEGAL_EARLY_FORECAST | MATR | N2-physics_basis | 169 | 123431 | 0.030426 | 0.051521 | 0.029962 | 0.015858 |

## 四项假设

### MATR

H1的GL配对MAE改善为0.007435，cell条件区间[0.005553, 0.009471]，相对容量对照改善为0.008536。当前配置支持显式特征增量。

H2的GL条件下rate辅助改善为-0.001836，cell条件区间[-0.003136, -0.000529]。该目标的稳定收益证据不足，详见正交交互与窗口反例。

H3的physics_free oracle MAE为0.001283，train-only共享尺度oracle为0.003006。预定0.01描述性阈值下，函数具备较低平均拟合误差，合法映射与优化仍需解释。

H4的free模式Jacobian条件数中位数为3.12e+16。近等价多解、profile范围与真实rate代理预测提供有限参数病态证据，无法证明仅凭K100的全局不可辨识。

### HUST

H1的GL配对MAE改善为0.002364，cell条件区间[0.000122, 0.004794]，相对容量对照改善为0.002403。当前配置支持显式特征增量。

H2的GL条件下rate辅助改善为-0.000984，cell条件区间[-0.002460, 0.000402]。该目标的稳定收益证据不足，详见正交交互与窗口反例。

H3的physics_free oracle MAE为0.000553，train-only共享尺度oracle为0.001005。预定0.01描述性阈值下，函数具备较低平均拟合误差，合法映射与优化仍需解释。

H4的free模式Jacobian条件数中位数为2.83e+07。近等价多解、profile范围与真实rate代理预测提供有限参数病态证据，无法证明仅凭K100的全局不可辨识。

## 验收计数

合法K100新增神经网络训练80组，正式SGD/AdamW更新126,000次，合法OOF评价246颗物理电池。D3个体多初值拟合问题2460个，每个8初值，共19680次个体优化。另有10个train-only共享尺度联合问题，共80次联合优化。D4 profile另有312次有限约束优化。rate proxy另有160个选定轻头与960次候选Ridge拟合。额外真实恢复烟测28次更新，原N012测试夹具的30次更新单列，均未纳入126000正式更新。

## 同输入、同辅助目标的解码头配对

| prediction_scope | domain | window | reference | candidate | n_cells | mean_improvement | median_improvement | better | worse | cell_ci_low | cell_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | MATR | all | D2-GL-rate-spline | D2-GL-rate-physics | 169 | -0.000131 | 0.000952 | 96 | 73 | -0.001407 | 0.001070 |
| LEGAL_EARLY_FORECAST | HUST | all | D2-GL-rate-spline | D2-GL-rate-physics | 77 | -0.001817 | -0.000265 | 35 | 42 | -0.004004 | 0.000265 |

该对比属于原合同要求，初版汇总遗漏其单独行，现从同一批冻结预测补齐。全部既有对比的估计量保持原定义。两头的结果用于判断具体三分量参数化的独立贡献，具有相同输入与辅助loss。

## 速率目标的时距权衡

| prediction_scope | domain | arm | window | supported_cells | cell_mean_mae | bias |
| --- | --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | 1001_plus | 77 | 0.038067 | 0.015332 |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | 501_1000 | 77 | 0.012972 | -0.005161 |
| LEGAL_EARLY_FORECAST | HUST | D1-GL | all | 77 | 0.021534 | 0.002299 |
| LEGAL_EARLY_FORECAST | HUST | D2-F1L1 | 1001_plus | 77 | 0.039222 | 0.009372 |
| LEGAL_EARLY_FORECAST | HUST | D2-F1L1 | 501_1000 | 77 | 0.013427 | -0.006074 |
| LEGAL_EARLY_FORECAST | HUST | D2-F1L1 | all | 77 | 0.022518 | -0.001070 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | 1001_plus | 40 | 0.079956 | 0.049368 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | 501_1000 | 151 | 0.052788 | 0.040725 |
| LEGAL_EARLY_FORECAST | MATR | D1-GL | all | 169 | 0.023226 | 0.008913 |
| LEGAL_EARLY_FORECAST | MATR | D2-F1L1 | 1001_plus | 40 | 0.076670 | 0.031270 |
| LEGAL_EARLY_FORECAST | MATR | D2-F1L1 | 501_1000 | 151 | 0.046018 | 0.029798 |
| LEGAL_EARLY_FORECAST | MATR | D2-F1L1 | all | 169 | 0.025062 | 0.002893 |

MATR的rate辅助使完整轨迹主指标上升，同时降低501–1000圈MAE与正偏差。当前结果支持目标之间的权衡，无法将固定0.1权重解释为历史M1优势的充分原因。HUST主指标同样没有出现该辅助项的稳定增益。

## 长随访贡献与反例

| prediction_scope | domain | window | reference | candidate | n_cells | positive_sum | top3_fraction_positive | largest_harm_cell | largest_harm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEGAL_EARLY_FORECAST | MATR | all | D1-0 | D1-GL | 169 | 1.412707 | 0.153585 | MATR_b2c43 | -0.027659 |
| LEGAL_EARLY_FORECAST | MATR | 1001_plus | D1-0 | D1-GL | 40 | 1.020604 | 0.368053 | MATR_b4c5 | -0.135329 |
| LEGAL_EARLY_FORECAST | HUST | all | D1-0 | D1-GL | 77 | 0.361246 | 0.284634 | HUST_1-6 | -0.017929 |
| LEGAL_EARLY_FORECAST | HUST | 1001_plus | D1-0 | D1-GL | 77 | 0.607357 | 0.249418 | HUST_1-6 | -0.036331 |

最大受损cell与top3正收益占比均保留，不能只依据总体均值解释所有电池。

## 速率代理、梯度与数值限制

MATR的200–500圈rate加入G统计后R²约0.658，500–1000圈只有42颗支持，当前各固定代理的R²约0.062–0.190。HUST的500–1000圈rate在原始低维统计下R²约0.561。可预测性随域和窗口变化，现有代理结果无法证明K100的全局不可预测性。

MATR训练probe中tau梯度范数中位数从第1轮约0.000218升至第12轮约0.007678，后续存在训练/验证分化。当前证据支持阶段性的弱激活与优化问题，初始化的因果贡献仍需独立控制。局部Jacobian病态与少量近等价初值应结合短随访支持解释。

MATR共156次profile优化，149次报告收敛，7次未收敛。未收敛点、约束残差及全部原始参数保留。

HUST共156次profile优化，147次报告收敛，9次未收敛。未收敛点、约束残差及全部原始参数保留。

保留局部分段物理表征与形状约束头的受控比较。当前三分量physics_basis的独立优势证据不足，固定query_mlp的2×2对照中rate=0.1未改善完整轨迹主指标。函数oracle的低误差支持继续检验参数映射和学习过程，不能直接认证真实电化学机制。

## 结论范围与后续

各域和时距的阴性结果与反例全部保留。本轮保持原问题、数据与80组预注册矩阵，不按测试结果扩模型。三分量形状先验仍可作为受控函数对照，具体LLI/LAM/SEI解释缺少电极分辨测量与独立分量真值。单种子与历史暴露限制外推，尚未满足论文候选升级要求。

优势为同队列、正交目标、显式信息预算、未来oracle隔离与实际权重重放。风险为有限oracle搜索、轻头样本量、近似参数匹配与多重探索。运行完整性置信度为高，跨域机制与训练种子稳定性置信度为低。

## 拓展思考

函数拟合能力与前缀映射需要各自的证据。保留实际可预测rate、参数病态与梯度覆盖之间的差异，有助于约束下一轮最小修改。

## 你可能想知道

全部逐cell预测、负例、断点状态、参数profile与源码哈希均有索引。HUST的右删失终点未被当作EOL事件。
