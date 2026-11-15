# M5 有界动力学与早期 Q(V) 可辨识性实验报告

执行日期为 2026-11-16。方法来源为 `research/m5-bounded-and-identifiability-20261116`，远端源提交为 `85c6ff92301a2e14f9ae72299f672f34a26bebe3`。父实验提交为 `565a638a52093891842b7278fec2c99d28258ccd`。原方法合同全文保留，合同 E 节记录的是远端编写时的执行状态，本报告给出本机实际执行结果。

## 研究问题与执行范围

真实循环 1–100 的共同 4C 放电 Q(V) 响应、已知快充策略及 SOH100，能否支持未来平均退化速率和阶段变化的预测，以及完整 SOH 轨迹外推。全部实验复用认证物理周期轴的 MATR 169 电池、原 44 个冻结数据文件、五个电池划分及一个未见协议划分。前缀电压 2.8–3.3 V、64 点，包含 20 维全局响应、64 点局部残差和 8×26 分段描述子。SOH 为原始放电容量除以 1.1 Ah。

唯一新深度模型为 `M5_BOUNDED_TRANSITION`。仅替换 M4 的动力学头，保留局部物理分解、动态持续性及两层真实 B 写入。独立可辨识性分析只使用冻结前缀与未来速率标签，两条实验流的选择互不回写。历史 M1、A1、M4 结果通过完整文件哈希核对后读取。

全部 169 个电池已在历史研究中暴露，本轮证据范围为单 seed42 的内部回顾性验证。完整轨迹按真实测得循环评分，随访外查询只提供数值诊断。当前没有真实认证 Knee-onset 事件标签。

## 实际训练与验收

| 项目 | 实际执行 |
| --- | --- |
| M5 正式训练 | 6 组，每组 150 epochs，共 12,450 次更新 |
| 额外完整复现 | cell_fold_0，重新从 seed42 初始化，150 epochs，2,100 次更新 |
| 工程 smoke | 1 epoch，14 次更新 |
| 总优化器更新 | 14,564 |
| 固定训练设置 | batch8，AdamW lr 0.0003，weight decay 0.01，float32，验证集 strict first-min |
| 模型参数 | 153,123，M4 原参数为 154,621 |
| 完整复现一致性 | 20/20 项 PASS，包含 150 轮核心日志、最终模型、优化器、RNG、选定权重及逐元素机制诊断 |
| 正常入口权重重放 | 6 个正式选定权重及 1 个复现权重，预测最大绝对误差 0 |
| M5 独立验收 | 28,771 项 PASS |
| 可辨识性独立验收 | 408 项 PASS，独立 NumPy 重算 200 个 Ridge 候选及 40 个选定模型 |
| 解码器工程检查 | CPU 39 项及 CUDA 89 项 PASS，CUDA 12 次真实首批前向、损失、全部梯度精确一致 |

验收核对了全部冻结输入、真实循环坐标、训练集专属标准化、每轮验证选权规则、AdamW 步数、两阶段速率端点、EOL5000、动态保持递推及 B 写入公式。解码头额外通过独立 NumPy 的相对强度、梯形积分、指数映射及绝对速率单位复核。逐轮未保存的中间权重无法独立重放，当前可核验记录为全部训练日志、最终优化器状态和选定权重。

### 相对上游代码的工程修正

- 将共享 M4 runner 的模型工厂、注册函数和元信息替换限制在可恢复的上下文中，避免同进程后续调用污染。
- 在有界趋势入口拒绝非有限或负的 SOH100 锚点，MATR 冻结队列全部通过。
- Ridge 验证集只选择 alpha。缺失值填充、标准化及最终系数均仅拟合该折训练集，取消上游脚本的 train+validation 重拟合。
- 保存全部 507 条 Hinge 资格记录，并补齐 skill、Spearman、协议宏平均、逐电池配对偏差及透明回归参数。
- Hinge 在固定位置时有 3 个线性系数，搜索转变位置再增加 1 个参数。保留条件 3 系数拟合比较，同时使用总参数数 4 对二次曲线 3 的 BIC 修正。
- 沿既有实验的验收方式，训练前登记了额外的一折完整复现，其 2,100 次更新单列计数。

模型、训练脚本及研究合同在正式运行期间保持哈希冻结。没有根据本轮测试成绩修改数据、窗口、模型、损失或重新选择超参数。

## 完整未来轨迹与历史参照

误差单位为 SOH fraction。主五折包含 169 个电池、77 个协议组、123,431 个真实未来观测。未见协议敏感性包含 46 个电池、15 个协议组、38,312 个观测。46 个电池属于原 169 个队列，两个 scope 应分别解释。

| scope | arm | trajectory_mae | trajectory_rmse | protocol_macro | supported_cells | supported_protocols | observations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main_oof | A1_M4_LEGACY_FLOW | 0.020148070 | 0.033650783 | 0.018174100 | 169 | 77 | 123431 |
| main_oof | M1 | 0.018761964 | 0.031578546 | 0.016988782 | 169 | 77 | 123431 |
| main_oof | M4_FULL_METHOD | 0.023271075 | 0.038527420 | 0.021528206 | 169 | 77 | 123431 |
| main_oof | M5_BOUNDED_TRANSITION | 0.021381156 | 0.036297182 | 0.018805205 | 169 | 77 | 123431 |
| unseen_protocol_sensitivity | A1_M4_LEGACY_FLOW | 0.027765230 | 0.042790148 | 0.026546788 | 46 | 15 | 38312 |
| unseen_protocol_sensitivity | M1 | 0.032187846 | 0.049509614 | 0.030318510 | 46 | 15 | 38312 |
| unseen_protocol_sensitivity | M4_FULL_METHOD | 0.048247783 | 0.074714973 | 0.047227677 | 46 | 15 | 38312 |
| unseen_protocol_sensitivity | M5_BOUNDED_TRANSITION | 0.030916580 | 0.049333177 | 0.029735647 | 46 | 15 | 38312 |

M5 相对 M4 的主五折 MAE 改善 8.12%，未见协议改善 35.92%。相对 M1 的主五折 MAE 增大 13.96%，相对 A1 增大 6.12%。未见协议相对 M1 的点估计改善 3.95%，相对 A1 增大 11.35%。配对结果保留两种重采样单位。

| scope | reference | estimator | improvement | ci95_low | ci95_high | improved_cells | worsened_cells | paired_cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_oof | M1 | cell_mean | -0.002619192 | -0.004118292 | -0.001100890 | 66 | 103 | 169 |
| main_oof | M1 | protocol_macro | -0.001816423 | -0.003385822 | -0.000012045 | 66 | 103 | 169 |
| main_oof | A1_M4_LEGACY_FLOW | cell_mean | -0.001233086 | -0.002926649 | 0.000533624 | 71 | 98 | 169 |
| main_oof | A1_M4_LEGACY_FLOW | protocol_macro | -0.000631106 | -0.002230386 | 0.001083047 | 71 | 98 | 169 |
| main_oof | M4_FULL_METHOD | cell_mean | 0.001889919 | 0.000835523 | 0.002987286 | 121 | 48 | 169 |
| main_oof | M4_FULL_METHOD | protocol_macro | 0.002723000 | 0.001583023 | 0.003877368 | 121 | 48 | 169 |
| unseen_protocol_sensitivity | M1 | cell_mean | 0.001271266 | -0.003227140 | 0.006003492 | 20 | 26 | 46 |
| unseen_protocol_sensitivity | M1 | protocol_macro | 0.000582863 | -0.004213142 | 0.007142239 | 20 | 26 | 46 |
| unseen_protocol_sensitivity | A1_M4_LEGACY_FLOW | cell_mean | -0.003151350 | -0.006628439 | 0.000505451 | 18 | 28 | 46 |
| unseen_protocol_sensitivity | A1_M4_LEGACY_FLOW | protocol_macro | -0.003188859 | -0.007300239 | 0.002087560 | 18 | 28 | 46 |
| unseen_protocol_sensitivity | M4_FULL_METHOD | cell_mean | 0.017331202 | 0.007051900 | 0.030275770 | 32 | 14 | 46 |
| unseen_protocol_sensitivity | M4_FULL_METHOD | protocol_macro | 0.017492030 | 0.005391496 | 0.035641296 | 32 | 14 | 46 |

`improvement` 为参照误差减 M5 误差，正值支持 M5。区间使用 2,000 次 seed42 配对 bootstrap，只刻画当前已拟合预测的电池或协议组抽样变异。相对 M4 的两个 scope、两个估计量区间均为正。主五折相对 M1 的两个区间均为负。相对 A1 及未见协议相对 M1 的区间覆盖零。跨训练种子变异尚未测量。

### 四时距分层

下表保留全部历史参照和各窗口实际支持分母。`1001_1000000` 是既有指标名，本队列真实最大观测循环为 2238。1001 圈后的轨迹支持为主五折 40 颗和未见协议 14 颗。两阶段速率标签的 42 颗共享支持还包括恰好观察到 1000 圈的电池。

| scope | arm | metric | cell_mean | protocol_macro | supported_cells | supported_protocols | observations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main_oof | A1_M4_LEGACY_FLOW | horizon_1001_1000000_mae | 0.070759972 | 0.066790562 | 40 | 19 | 10614 |
| main_oof | A1_M4_LEGACY_FLOW | horizon_101_200_mae | 0.002494665 | 0.003546346 | 169 | 77 | 16865 |
| main_oof | A1_M4_LEGACY_FLOW | horizon_201_500_mae | 0.007441912 | 0.011273854 | 168 | 76 | 49795 |
| main_oof | A1_M4_LEGACY_FLOW | horizon_501_1000_mae | 0.040369811 | 0.040225235 | 151 | 61 | 46157 |
| main_oof | M1 | horizon_1001_1000000_mae | 0.063976380 | 0.064719301 | 40 | 19 | 10614 |
| main_oof | M1 | horizon_101_200_mae | 0.002532852 | 0.003473166 | 169 | 77 | 16865 |
| main_oof | M1 | horizon_201_500_mae | 0.007562969 | 0.011339646 | 168 | 76 | 49795 |
| main_oof | M1 | horizon_501_1000_mae | 0.036761921 | 0.033654548 | 151 | 61 | 46157 |
| main_oof | M4_FULL_METHOD | horizon_1001_1000000_mae | 0.082963804 | 0.085623503 | 40 | 19 | 10614 |
| main_oof | M4_FULL_METHOD | horizon_101_200_mae | 0.005372106 | 0.007926597 | 169 | 77 | 16865 |
| main_oof | M4_FULL_METHOD | horizon_201_500_mae | 0.010124415 | 0.014615633 | 168 | 76 | 49795 |
| main_oof | M4_FULL_METHOD | horizon_501_1000_mae | 0.045065565 | 0.044656086 | 151 | 61 | 46157 |
| main_oof | M5_BOUNDED_TRANSITION | horizon_1001_1000000_mae | 0.069207992 | 0.069003481 | 40 | 19 | 10614 |
| main_oof | M5_BOUNDED_TRANSITION | horizon_101_200_mae | 0.004242297 | 0.006146778 | 169 | 77 | 16865 |
| main_oof | M5_BOUNDED_TRANSITION | horizon_201_500_mae | 0.008215750 | 0.011843146 | 168 | 76 | 49795 |
| main_oof | M5_BOUNDED_TRANSITION | horizon_501_1000_mae | 0.042543195 | 0.040628354 | 151 | 61 | 46157 |
| unseen_protocol_sensitivity | A1_M4_LEGACY_FLOW | horizon_1001_1000000_mae | 0.111940705 | 0.125972519 | 14 | 6 | 6267 |
| unseen_protocol_sensitivity | A1_M4_LEGACY_FLOW | horizon_101_200_mae | 0.003511575 | 0.006042808 | 46 | 15 | 4571 |
| unseen_protocol_sensitivity | A1_M4_LEGACY_FLOW | horizon_201_500_mae | 0.008064308 | 0.012731941 | 45 | 14 | 13378 |
| unseen_protocol_sensitivity | A1_M4_LEGACY_FLOW | horizon_501_1000_mae | 0.044338109 | 0.046067209 | 40 | 11 | 14096 |
| unseen_protocol_sensitivity | M1 | horizon_1001_1000000_mae | 0.127810673 | 0.134262451 | 14 | 6 | 6267 |
| unseen_protocol_sensitivity | M1 | horizon_101_200_mae | 0.004042415 | 0.006686631 | 46 | 15 | 4571 |
| unseen_protocol_sensitivity | M1 | horizon_201_500_mae | 0.009171886 | 0.014656530 | 45 | 14 | 13378 |
| unseen_protocol_sensitivity | M1 | horizon_501_1000_mae | 0.049740205 | 0.051826740 | 40 | 11 | 14096 |
| unseen_protocol_sensitivity | M4_FULL_METHOD | horizon_1001_1000000_mae | 0.209307770 | 0.183884133 | 14 | 6 | 6267 |
| unseen_protocol_sensitivity | M4_FULL_METHOD | horizon_101_200_mae | 0.005982655 | 0.011213822 | 46 | 15 | 4571 |
| unseen_protocol_sensitivity | M4_FULL_METHOD | horizon_201_500_mae | 0.014097646 | 0.024097198 | 45 | 14 | 13378 |
| unseen_protocol_sensitivity | M4_FULL_METHOD | horizon_501_1000_mae | 0.050433447 | 0.059464426 | 40 | 11 | 14096 |
| unseen_protocol_sensitivity | M5_BOUNDED_TRANSITION | horizon_1001_1000000_mae | 0.137481217 | 0.141854774 | 14 | 6 | 6267 |
| unseen_protocol_sensitivity | M5_BOUNDED_TRANSITION | horizon_101_200_mae | 0.004889540 | 0.008949513 | 46 | 15 | 4571 |
| unseen_protocol_sensitivity | M5_BOUNDED_TRANSITION | horizon_201_500_mae | 0.008329590 | 0.014547352 | 45 | 14 | 13378 |
| unseen_protocol_sensitivity | M5_BOUNDED_TRANSITION | horizon_501_1000_mae | 0.042414943 | 0.045226063 | 40 | 11 | 14096 |

各时距 RMSE、逐折误差、逐协议误差和逐电池误差同时保存在 `metric_summary.csv`、`fold_metrics.csv`、`protocol_metrics.csv`、`cell_metrics.csv`。M5 的远期窗口误差明显高于早期窗口，主五折总体均值会受到不同随访长度的影响。

### 区间速率与 EOL

M5 区间速率由最终 SOH 预测的两个真实周期端点差除以周期差计算，单位为 SOH/cycle。原两个辅助损失保持原设置。

| scope | metric | cell_mean | protocol_macro | supported_cells | supported_protocols |
| --- | --- | --- | --- | --- | --- |
| main_oof | r200_500_absolute_error | 0.0000579569 | 0.0000772717 | 151 | 61 |
| main_oof | r500_1000_absolute_error | 0.0001237499 | 0.0001231879 | 42 | 19 |
| unseen_protocol_sensitivity | r200_500_absolute_error | 0.0000566243 | 0.0000826133 | 40 | 11 |
| unseen_protocol_sensitivity | r500_1000_absolute_error | 0.0001500273 | 0.0001752980 | 14 | 6 |

EOL 使用潜在 trend 在 80% SOH 的第一次 crossing，固定查询上限为 5000 圈。观察到的 EOL 事件和右删失状态来自冻结元数据。

| scope | arm | harrell_c | comparable_pairs | executed_test_cells | observed_events | right_censored | predicted_no_crossing_through5000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main_oof | M5_BOUNDED_TRANSITION | 0.840186191 | 14179 | 169 | 169 | 0 | 0 |
| main_oof | M1 | 0.861132661 | 14179 | 169 | 169 | 0 | 1 |
| main_oof | A1_M4_LEGACY_FLOW | 0.851399958 | 14179 | 169 | 169 | 0 | 1 |
| main_oof | M4_FULL_METHOD | 0.826715565 | 14179 | 169 | 169 | 0 | 0 |
| unseen_protocol_sensitivity | M5_BOUNDED_TRANSITION | 0.828488372 | 1032 | 46 | 46 | 0 | 0 |
| unseen_protocol_sensitivity | M1 | 0.838178295 | 1032 | 46 | 46 | 0 | 0 |
| unseen_protocol_sensitivity | A1_M4_LEGACY_FLOW | 0.838178295 | 1032 | 46 | 46 | 0 | 0 |
| unseen_protocol_sensitivity | M4_FULL_METHOD | 0.804263566 | 1032 | 46 | 46 | 0 | 0 |

| scope | reference | sampling_unit | improvement | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- |
| main_oof | M1 | cell | -0.020946470 | -0.041349945 | -0.002696910 |
| main_oof | A1_M4_LEGACY_FLOW | cell | -0.011213767 | -0.030198137 | 0.005745345 |
| main_oof | M4_FULL_METHOD | cell | 0.013470626 | -0.004119479 | 0.030337615 |
| main_oof | M1 | protocol_group | -0.020946470 | -0.040611204 | -0.002606232 |
| main_oof | A1_M4_LEGACY_FLOW | protocol_group | -0.011213767 | -0.027747818 | 0.004694630 |
| main_oof | M4_FULL_METHOD | protocol_group | 0.013470626 | -0.004450621 | 0.034409993 |
| unseen_protocol_sensitivity | M1 | cell | -0.009689922 | -0.061526562 | 0.035874191 |
| unseen_protocol_sensitivity | A1_M4_LEGACY_FLOW | cell | -0.009689922 | -0.056161797 | 0.030815876 |
| unseen_protocol_sensitivity | M4_FULL_METHOD | cell | 0.024224806 | -0.031685528 | 0.088037480 |
| unseen_protocol_sensitivity | M1 | protocol_group | -0.009689922 | -0.042417303 | 0.058828636 |
| unseen_protocol_sensitivity | A1_M4_LEGACY_FLOW | protocol_group | -0.009689922 | -0.048616038 | 0.047934646 |
| unseen_protocol_sensitivity | M4_FULL_METHOD | protocol_group | 0.024224806 | -0.036192014 | 0.076698391 |

EOL 的 `improvement` 为 M5 C-index 减参照 C-index，仍采用相同拟合预测上的条件重采样。M5 的 EOL 点估计处于 M4 与 M1/A1 之间。

### 已知反例 MATR_b1c2

保留未见协议中的原反例，其 2,136 个未来真实观测全部纳入。

| arm | trajectory_mae | trajectory_rmse |
| --- | --- | --- |
| A1_M4_LEGACY_FLOW | 0.078286818 | 0.099161165 |
| M1 | 0.091737269 | 0.123257311 |
| M4_FULL_METHOD | 0.277551195 | 0.411164066 |
| M5_BOUNDED_TRANSITION | 0.095258370 | 0.137446018 |

M5 在此电池改善了 M4 的大误差，MAE 仍高于 M1 与 A1。该电池没有被剔除或用于修改本轮结构。

## 有界趋势、观测残差与软转变参数

内部相对容量损失强度 λ 的固定上界为每千真实循环 2。时间变量 τ=(k−100)/1000，潜在趋势为 SOH100×exp(−H)，H 为 λ 关于 τ 的固定整数周期梯形积分。`integrated_loss` 输出表示 SOH100−trend，`relative_hazard_integral` 才表示 H。对外绝对损失速率为 trend×λ，单位为 SOH/1000cycles，再除以 1000 得到 SOH/cycle。此 λ 描述容量相对损失强度。

M5 潜在趋势具有非负、非增的结构边界。观测输出叠加原 0.03×tanh 残差，再裁剪至 [0,1.2]。本轮真实未来观测上的潜在趋势及最终预测回升次数均为 0，最终预测的结构仍允许残差引起局部回升。

| scope | query_region | query_count | raw_trend_min | raw_trend_below_zero_count | trend_clipped_count | observation_clipped_count | observation_clipped_affected_cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main_oof | all_queries_100_5000 | 828269 | 0.000276733 | 0 | 0 | 4251 | 2 |
| main_oof | beyond_cell_last_followup | 704630 | 0.000276733 | 0 | 0 | 4251 | 2 |
| main_oof | observed_future | 123431 | 0.496388465 | 0 | 0 | 0 | 0 |
| unseen_protocol_sensitivity | all_queries_100_5000 | 225446 | 0.000284763 | 0 | 0 | 1154 | 2 |
| unseen_protocol_sensitivity | beyond_cell_last_followup | 187079 | 0.000284763 | 0 | 0 | 1154 | 2 |
| unseen_protocol_sensitivity | observed_future | 38312 | 0.531628370 | 0 | 0 | 0 | 0 |

全部查询与随访外查询的潜在趋势负值、趋势裁剪均为 0。随访外的最终观测输出仍有下界裁剪，主五折 4,251 点涉及 2 个电池，未见协议 1,154 点涉及 2 个电池。两个 scope 存在重叠，数量分别报告。上述点均位于随访范围以外，无法用真实 SOH 认证其准确性。

| scope | parameter | minimum | median | maximum |
| --- | --- | --- | --- | --- |
| main_oof | soft_transition_center_cycle | 309.005615 | 552.857971 | 1692.806641 |
| main_oof | soft_transition_width_cycles | 50.365070 | 56.467003 | 196.140732 |
| main_oof | soft_transition_signed_amplitude | 1.066166 | 1.994677 | 1.999923 |
| main_oof | relative_intensity_min_per1000 | 0.028056 | 0.057615 | 0.334732 |
| main_oof | relative_intensity_max_per1000 | 0.244959 | 1.417457 | 1.886880 |
| unseen_protocol_sensitivity | soft_transition_center_cycle | 317.648438 | 575.416870 | 1091.473633 |
| unseen_protocol_sensitivity | soft_transition_width_cycles | 53.867279 | 65.282722 | 115.620461 |
| unseen_protocol_sensitivity | soft_transition_signed_amplitude | 1.715112 | 1.980260 | 1.999384 |
| unseen_protocol_sensitivity | relative_intensity_min_per1000 | 0.029699 | 0.046248 | 0.307972 |
| unseen_protocol_sensitivity | relative_intensity_max_per1000 | 0.725708 | 1.292744 | 1.875772 |

主五折软转变中心中位数约 552.86 圈，宽度中位数 56.47 圈，有符号幅度中位数 1.99468。宽度与幅度接近各自允许边界，记录为拟合参数分布。它们尚缺跨种子或参数剖面的唯一性验证。此处的 cycle center 仅具有模型坐标意义，真实 Knee 认证事件数为 0。

## 独立早期 Q(V) 速率可辨识性分析

C0 包含 15 维策略值、15 维可用掩码及 SOH100，共 31 维。C1 再加入最近合格且真实循环≤100 的 20 维全局响应、8 个电压段的 signed mean/RMS 共 16 维、36 维对应掩码，共 103 维。逐电池特征截止与支持见 `early_feature_audit.csv`。

原五折每项任务、每个 C0/C1 各选一个 Ridge，共 40 个选定模型。固定 alpha 候选为 [0.01,0.1,1,10,100]，总 200 次候选拟合。目标在拟合时乘 1000，输出恢复为 SOH/真实循环。验证集 strict first-min，全部最终系数保持训练集拟合。

skill=1−OOF模型MAE/OOF训练均值预测MAE，训练均值由每折训练标签单独计算。正 skill 表示优于这一基线。下表包含真实周期单位 MAE、协议宏平均及排序相关。

| task | model | unique_test_cells | supported_protocols | mae | skill_vs_training_mean | spearman | protocol_macro_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| change_late_minus_mid_shared | C0 | 42 | 19 | 0.0001388270 | 0.1179723203 | 0.5920914026 | 0.0001367765 |
| change_late_minus_mid_shared | C1 | 42 | 19 | 0.0001431943 | 0.0902249141 | 0.5760473219 | 0.0001462531 |
| level_late_shared | C0 | 42 | 19 | 0.0001420616 | 0.1149931240 | 0.5894984199 | 0.0001401561 |
| level_late_shared | C1 | 42 | 19 | 0.0001446168 | 0.0990752976 | 0.5869054372 | 0.0001483818 |
| level_mid_all | C0 | 151 | 61 | 0.0001453007 | 0.1848312227 | 0.3786023005 | 0.0001720763 |
| level_mid_all | C1 | 151 | 61 | 0.0000917492 | 0.4852670883 | 0.7959079819 | 0.0001167555 |
| level_mid_shared | C0 | 42 | 19 | 0.0000050924 | 0.1401880387 | 0.6505955757 | 0.0000051514 |
| level_mid_shared | C1 | 42 | 19 | 0.0000056951 | 0.0384222628 | 0.4593630986 | 0.0000061521 |

| task | sampling_unit | paired_cells | protocol_groups | improvement | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| change_late_minus_mid_shared | cell | 42 | 19 | -0.0000043673 | -0.0000263567 | 0.0000175048 |
| change_late_minus_mid_shared | protocol_group | 42 | 19 | -0.0000094765 | -0.0000347427 | 0.0000172573 |
| level_late_shared | cell | 42 | 19 | -0.0000025551 | -0.0000253902 | 0.0000201856 |
| level_late_shared | protocol_group | 42 | 19 | -0.0000082257 | -0.0000349625 | 0.0000196749 |
| level_mid_all | cell | 151 | 61 | 0.0000535515 | 0.0000376120 | 0.0000688893 |
| level_mid_all | protocol_group | 151 | 61 | 0.0000553207 | 0.0000243687 | 0.0000815242 |
| level_mid_shared | cell | 42 | 19 | -0.0000006027 | -0.0000015443 | 0.0000002802 |
| level_mid_shared | protocol_group | 42 | 19 | -0.0000010007 | -0.0000020806 | 0.0000000755 |

配对 `improvement` 为 C0 绝对误差减 C1 绝对误差，正值支持额外 Q(V) 特征。全支持的中期速率 151 颗中，C1 将 skill 从约 0.185 提升到 0.485，Spearman 从约 0.379 提升到 0.796，电池及协议组配对区间均为正。共享 42 颗中的中期、晚期及变化量任务，C1 的误差点估计均增大，配对区间均覆盖零。

共享队列 42 颗全部具有正的晚期减中期速率，晚期/中期比中位数为 4.786500，跨阶段 Pearson 为 0.574793。这些统计反映已随访到 1000 圈的选择性样本。C0 对变化量本身已具有约 0.118 的正 skill 和约 0.592 的 Spearman，变化量的符号无法单独证明预测能力。现有特征及 Ridge 模型控制已知策略和 SOH100 后，对变化量的额外信息证据不足，物理可辨识性仍需更充分证据。

## Hinge 候选与二次光滑对照

此诊断使用未来已观测轨迹。三个拟合起点为 101、201、301 圈，至少 240 个观测且跨度至少 300 圈。最多保留 500 个原生坐标点，使用 20%–80% 坐标段的 75 个 κ 候选。5% SSE 剖面另加 1e−12 数值余量。全部 169×3=507 条资格记录保留，其中 434 条合格、73 条支持不足。

| fit_start_cycle | total_cells | fitted | insufficient | median_kappa | median_profile_width | median_corrected_BIC | positive_corrected_BIC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 169 | 166 | 3 | 634.500000 | 10.000000 | 857.994232 | 166 |
| 201 | 169 | 151 | 18 | 666.000000 | 11.000000 | 784.489036 | 150 |
| 301 | 169 | 117 | 52 | 716.000000 | 11.000000 | 737.959381 | 114 |

同 117 个在三个起点均合格的电池中，κ 最大值减最小值的中位数为 15.0 圈，最大值为 40.0 圈，超过 100 圈的电池数为 0。这些轨迹的候选位置在所测起点变化下较稳定。合格轨迹中 Hinge 的拟合优势与起点稳定性，仍不足以认证真实物理 Knee 时机。

固定 κ 的 Hinge 有 3 个线性系数，搜索 κ 后共 4 个参数。二次曲线有 3 个参数。原合同的同自由度描述对应条件拟合比较，整体模型选择额外报告修正后的 BIC_quad−BIC_free_hinge=n×log(SSE_quad/SSE_hinge)−log(n)。正值支持自由 Hinge。零附近 SSE 使用预登记的 1e−16 数值下限。

光滑二次函数与光滑指数函数各设置三个起点，共 6 个无真实 Knee 的合成负对照。

| synthetic_source | fit_start_cycle | kappa_proxy | profile_width_5pct_SSE | bic_quad_minus_free_hinge |
| --- | --- | --- | --- | --- |
| smooth_quadratic | 101 | 1052.000000 | 95.000000 | -16024.766356 |
| smooth_quadratic | 201 | 1102.000000 | 91.000000 | -15916.489357 |
| smooth_quadratic | 301 | 1152.000000 | 85.000000 | -15802.231598 |
| smooth_exponential | 101 | 1018.000000 | 96.000000 | -2344.991818 |
| smooth_exponential | 201 | 1070.000000 | 90.000000 | -2399.172360 |
| smooth_exponential | 301 | 1135.000000 | 85.000000 | -2456.411764 |

两类光滑负对照均能产出具体 Hinge 坐标，修正后的 BIC 均支持二次光滑拟合。候选坐标的出现本身缺少事件认证效力。候选参数与 M5 soft center 均未进入早期速率回归输入或监督。本轮没有真实 Knee 时机精准率。

## 优势、劣势与置信度

| 视角 | 优势 | 劣势与风险 | 置信度及依据 |
| --- | --- | --- | --- |
| 数值动力学 | 锚点精确，潜在趋势非负非增，查询至 5000 圈保持有限，随访内裁剪为 0 | 随访外观测残差仍可触发裁剪，拟合幅度与宽度接近边界 | 高，公式独立验算、7 个权重重放和逐点诊断一致 |
| 预测评价 | 相对 M4 的两个 scope、两种配对估计均支持改善 | 主五折弱于 M1，相对 A1 点估计较差，远期 SOH 误差仍高 | 中，完整单 seed 内部回顾性结果，历史测试暴露 |
| 信息与物理解释 | 151 颗队列中发现早期 Q(V) 对中期速率的增量 | 共享 42 颗的 Δr 增量不足，随访选择、模型容量及特征表达均会影响结果 | 中，限定于当前特征、Ridge 和队列 |
| 机制及外部外推 | 可完整追踪拟合参数和反例 | 单种子、无独立真实 Knee、无新外部队列，参数唯一性未验 | 低，当前交付缺少对应独立验证 |

## 拓展思考

本轮同时保留了数值约束的收益和总体预测误差的代价。解码器的结构可行性、早期特征对速率水平的增量、阶段变化量的增量以及软转变参数的物理认证，分别需要对应证据。当前结果支持缩小科学主张的范围，原固定实验已全部完成。

## 你可能想知道

- 169、151、42、40 分别对应完整队列、中期端点速率支持、两个阶段共同端点支持、1000 圈之后仍有观测的轨迹支持。它们具有不同评价分母。
- 主五折与未见协议使用相同原队列中的固定划分，46 个协议测试电池已有主 OOF 预测，两个 scope 的误差应分别报告。
- 软转变中心与 Hinge κ 都使用 cycle 作为坐标，其可重复拟合尚不能完成物理事件标注。
- 置信区间均条件于已拟合预测。单 seed42 及一折完整复现提供当前环境的重复性检查，跨种子稳健性仍无实测结果。

## 产物与复核入口

完整 M5 结果位于 `results/unified_matr_m5_bounded_seed42/`，独立速率分析位于 `results/unified_matr_m5_identifiability_seed42/`。六组正式训练、一个完整复现的选定权重和最终训练状态、机制 NPZ、逐电池预测、训练日志及验收 JSON 一并交付。原始数据保持只读，原有科研文件保持字节一致，README 增加本轮入口。

主要验收文件为 `results/unified_matr_m5_bounded_seed42/RUN_VERIFICATION.json` 和 `results/unified_matr_m5_identifiability_seed42/RUN_VERIFICATION.json`。模型注册 SHA256 为 `47c4432fbba2dcfa4ef528f05227adbf0e83211bd59b0e2171f597b9a507a874`。数据冻结 SHA256 为 `276016ea2f72c2d9c366c44b00906c80cf8234f741b48b60a6a2562c57fff4fb`。

```powershell
python scripts/unified_matr/train_m5.py --data data/unified_matr_seed42 --output-root results/unified_matr_m5_bounded_seed42 --methodology-contract research/M5_BOUNDED_AND_RATE_IDENTIFIABILITY_CONTRACT_ZH.md --mode register --device cuda
python scripts/unified_matr/train_m5.py --data data/unified_matr_seed42 --output-root results/unified_matr_m5_bounded_seed42 --methodology-contract research/M5_BOUNDED_AND_RATE_IDENTIFIABILITY_CONTRACT_ZH.md --mode smoke --device cuda
python scripts/unified_matr/train_m5.py --data data/unified_matr_seed42 --output-root results/unified_matr_m5_bounded_seed42 --methodology-contract research/M5_BOUNDED_AND_RATE_IDENTIFIABILITY_CONTRACT_ZH.md --mode train --device cuda
python scripts/unified_matr/train_m5.py --data data/unified_matr_seed42 --output-root results/unified_matr_m5_bounded_seed42 --methodology-contract research/M5_BOUNDED_AND_RATE_IDENTIFIABILITY_CONTRACT_ZH.md --mode reproduce --device cuda
python scripts/unified_matr/analyze_m5_identifiability.py --data data/unified_matr_seed42 --output results/unified_matr_m5_identifiability_seed42
python scripts/verify_m5_identifiability.py --data data/unified_matr_seed42 --output results/unified_matr_m5_identifiability_seed42
python scripts/analyze_m5_results.py --data data/unified_matr_seed42 --output results/unified_matr_m5_bounded_seed42 --device cuda
python scripts/verify_m5_results.py --data data/unified_matr_seed42 --output results/unified_matr_m5_bounded_seed42 --device cuda
```

本机实际 Python 为 `D:/Anaconda/envs/tslib/python.exe`，Python 3.11.14，torch 2.4.1+cu121，CUDA 12.1，GPU 为 RTX 4060 Laptop 8 GiB。单 GPU 串行执行训练与权重重放。
