# A–C 源工况诊断实验结果

## 执行范围

本报告覆盖A。新增真实神经训练27项，复用已完成冻结训练27项，新增优化更新25920次，独立重算108项外层结果与20700条电芯窗口误差。最终目标Query未来标签使用量为0。

源池为115颗物理电芯、62个工况组，沿用三折与42、123、2026三个种子。源池和工况划分存在历史研究暴露，各折训练集重叠，当前证据属于回顾性源内部诊断。

## 模型完整未来与远期误差

| study | regime | mode | evaluation_mode | window | mae | physical_cells | protocol_groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | frozen | absolute | matched_norm | 1001_plus | 0.095834 | 21 | 13 |
| A | frozen | absolute | matched_norm | all | 0.033021 | 115 | 62 |
| A | frozen | absolute | trained_ball | 1001_plus | 0.100473 | 21 | 13 |
| A | frozen | absolute | trained_ball | all | 0.025718 | 115 | 62 |
| A | frozen | constant | matched_norm | 1001_plus | 0.097646 | 21 | 13 |
| A | frozen | constant | matched_norm | all | 0.033528 | 115 | 62 |
| A | frozen | constant | trained_ball | 1001_plus | 0.100533 | 21 | 13 |
| A | frozen | constant | trained_ball | all | 0.025744 | 115 | 62 |
| A | frozen | dual | matched_norm | 1001_plus | 0.097665 | 21 | 13 |
| A | frozen | dual | matched_norm | all | 0.033083 | 115 | 62 |
| A | frozen | dual | trained_ball | 1001_plus | 0.100464 | 21 | 13 |
| A | frozen | dual | trained_ball | all | 0.025701 | 115 | 62 |
| A | joint | absolute | matched_norm | 1001_plus | 0.091690 | 21 | 13 |
| A | joint | absolute | matched_norm | all | 0.025944 | 115 | 62 |
| A | joint | absolute | trained_ball | 1001_plus | 0.088546 | 21 | 13 |
| A | joint | absolute | trained_ball | all | 0.025156 | 115 | 62 |
| A | joint | constant | matched_norm | 1001_plus | 0.090709 | 21 | 13 |
| A | joint | constant | matched_norm | all | 0.025560 | 115 | 62 |
| A | joint | constant | trained_ball | 1001_plus | 0.090256 | 21 | 13 |
| A | joint | constant | trained_ball | all | 0.024981 | 115 | 62 |
| A | joint | dual | matched_norm | 1001_plus | 0.091957 | 21 | 13 |
| A | joint | dual | matched_norm | all | 0.025712 | 115 | 62 |
| A | joint | dual | trained_ball | 1001_plus | 0.091370 | 21 | 13 |
| A | joint | dual | trained_ball | all | 0.024937 | 115 | 62 |

## A 门控与联合优化的交互

所有组从同折同种子的相同静态C0与Bridge/Head参数出发。适配核心和门参数均为1440，joint额外训练相同的63169个Bridge/Head参数。冻结与联合组使用相同训练轮数、选权方式、损失和实际适配权重增量半径。

下表的收益为对应constant的MAE减去候选MAE。交互项为joint门控收益减去frozen门控收益，正值对应联合优化增强门控收益。

| 对照 | 组 | 评价 | 窗口 | 候选 | 参照 | MAE收益 | 改善% | 电芯CI | 工况CI | 各折均正 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gating_within_regime | frozen | trained_ball | all | absolute | constant | 0.000026 | 0.101089 | [-0.0001211721176053773, 0.00016698747054043204] | [-0.00011820779474107037, 0.00011513847252444272] | False |
| gating_within_regime | frozen | trained_ball | all | dual | constant | 0.000043 | 0.167132 | [-9.379281001661907e-05, 0.0001611855168354658] | [-6.115326820773101e-05, 0.00017208246163326392] | False |
| gating_within_regime | frozen | trained_ball | 1001_plus | absolute | constant | 0.000060 | 0.059378 | [-0.0009304172058147467, 0.0012549399659569705] | [-0.0013058128148968513, 0.0015678011881802305] | False |
| gating_within_regime | frozen | trained_ball | 1001_plus | dual | constant | 0.000069 | 0.068323 | [-0.0013451146192315219, 0.0016672263735970135] | [-0.001410828343736655, 0.0025808764751225426] | False |
| gating_within_regime | frozen | matched_norm | all | absolute | constant | 0.000507 | 1.513296 | [-0.00024008922315737897, 0.0016462051187073007] | [-0.00016676533950601345, 0.0011864098806832325] | False |
| gating_within_regime | frozen | matched_norm | all | dual | constant | 0.000445 | 1.328330 | [-0.00030263138869115547, 0.0013334250917129168] | [-0.0005417248800509042, 0.001022338369638359] | False |
| gating_within_regime | frozen | matched_norm | 1001_plus | absolute | constant | 0.001812 | 1.855451 | [-0.0028337596523675054, 0.009882914947822796] | [-0.003840246545849901, 0.011992015589181124] | True |
| gating_within_regime | frozen | matched_norm | 1001_plus | dual | constant | -0.000020 | -0.020290 | [-0.007195989404904417, 0.009536324278865643] | [-0.009337777064514118, 0.014122768138025524] | False |
| gating_within_regime | joint | trained_ball | all | absolute | constant | -0.000175 | -0.702477 | [-0.0010661433539340274, 0.0006856507528021444] | [-0.0010830274420532712, 0.0005941414502222407] | False |
| gating_within_regime | joint | trained_ball | all | dual | constant | 0.000044 | 0.174466 | [-0.0005483732150974846, 0.0006610318859897661] | [-0.0008069343865082663, 0.0005557177430310452] | False |
| gating_within_regime | joint | trained_ball | 1001_plus | absolute | constant | 0.001711 | 1.895551 | [-0.006388571454771862, 0.00927866161988218] | [-0.0036264129073825175, 0.01019461510774174] | False |
| gating_within_regime | joint | trained_ball | 1001_plus | dual | constant | -0.001113 | -1.233640 | [-0.0077833197188492045, 0.005311287141080723] | [-0.006849349806564385, 0.006755661629999768] | False |
| gating_within_regime | joint | matched_norm | all | absolute | constant | -0.000384 | -1.503101 | [-0.0016735562189535641, 0.0010309752717158004] | [-0.0016010684765879613, 0.0008352212644736495] | False |
| gating_within_regime | joint | matched_norm | all | dual | constant | -0.000152 | -0.593777 | [-0.0008688021358203016, 0.0006291350011155517] | [-0.0011083882283154964, 0.000581564182841275] | False |
| gating_within_regime | joint | matched_norm | 1001_plus | absolute | constant | -0.000982 | -1.082048 | [-0.010362591983074518, 0.008091179868259694] | [-0.00844867120509901, 0.01030319246880507] | False |
| gating_within_regime | joint | matched_norm | 1001_plus | dual | constant | -0.001248 | -1.375788 | [-0.007773567192203751, 0.0064445327452428034] | [-0.006804968238739024, 0.009708175506924058] | False |
| joint_gate_gain_minus_frozen_gate_gain | interaction | trained_ball | all | absolute | constant | -0.000202 | — | [-0.0010939630427905076, 0.0006145483841178989] | [-0.0010067505392559269, 0.0005676102335717256] | False |
| joint_gate_gain_minus_frozen_gate_gain | interaction | trained_ball | all | dual | constant | 0.000001 | — | [-0.0006289753399616226, 0.0006129692992727166] | [-0.0009199504126814348, 0.000505482185215793] | False |
| joint_gate_gain_minus_frozen_gate_gain | interaction | trained_ball | 1001_plus | absolute | constant | 0.001651 | — | [-0.007187348555368545, 0.00940678832908433] | [-0.003826588161045493, 0.009345827836371029] | False |
| joint_gate_gain_minus_frozen_gate_gain | interaction | trained_ball | 1001_plus | dual | constant | -0.001182 | — | [-0.008116842341065982, 0.0051782315955428105] | [-0.007342473206361947, 0.006684612587420187] | False |
| joint_gate_gain_minus_frozen_gate_gain | interaction | matched_norm | all | absolute | constant | -0.000892 | — | [-0.0028324469558188687, 0.0010133301797667288] | [-0.001867615489190038, 0.0004932178266254697] | False |
| joint_gate_gain_minus_frozen_gate_gain | interaction | matched_norm | all | dual | constant | -0.000597 | — | [-0.00157069990244055, 0.0002088738508004398] | [-0.0012245137288106804, 0.00045337600396026226] | False |
| joint_gate_gain_minus_frozen_gate_gain | interaction | matched_norm | 1001_plus | absolute | constant | -0.002793 | — | [-0.01757287930728241, 0.008321359410349257] | [-0.017320774089267278, 0.010323137006547276] | False |
| joint_gate_gain_minus_frozen_gate_gain | interaction | matched_norm | 1001_plus | dual | constant | -0.001228 | — | [-0.013247841668086003, 0.009181642906709442] | [-0.01826357923664291, 0.013205367743709145] | False |

## 完整未来与远期对照的逐折与逐种子方向

| 实验 | 模型或组 | 评价 | 窗口 | 候选 | 参照 | 各折MAE收益 | 各种子MAE收益 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | frozen | trained_ball | all | absolute | constant | 0=-5.58919e-05 / 1=+0.0001189632 / 2=+1.715681e-05 | 42=+2.875766e-05 / 123=-3.856381e-05 / 2026=+8.787853e-05 |
| A | frozen | trained_ball | all | dual | constant | 0=-3.669735e-05 / 1=+0.0001335032 / 2=+3.43702e-05 | 42=+2.446479e-05 / 123=-3.064222e-06 / 2026=+0.0001076775 |
| A | frozen | trained_ball | 1001_plus | absolute | constant | 0=+0.0009655713 / 1=+0.0002047782 / 2=-0.0009912658 | 42=+9.577391e-05 / 123=+0.0003549485 / 2026=-0.0002716386 |
| A | frozen | trained_ball | 1001_plus | dual | constant | 0=+0.001491906 / 1=-0.0003442008 / 2=-0.0009416438 | 42=-7.068778e-05 / 123=+0.0008609437 / 2026=-0.0005841943 |
| A | frozen | matched_norm | all | absolute | constant | 0=-0.0005305463 / 1=+0.0001373486 / 2=+0.001942653 | 42=-0.0001676534 / 123=+0.001484483 / 2026=+0.000205312 |
| A | frozen | matched_norm | all | dual | constant | 0=-0.0001741296 / 1=-0.0005288021 / 2=+0.002055328 | 42=+0.0005230409 / 123=+0.0006734912 / 2026=+0.0001395621 |
| A | frozen | matched_norm | 1001_plus | absolute | constant | 0=+0.0009516691 / 1=+0.001331139 / 2=+0.003152489 | 42=+0.0003045996 / 123=+0.006621826 / 2026=-0.001491129 |
| A | frozen | matched_norm | 1001_plus | dual | constant | 0=+0.001445535 / 1=-0.004984892 / 2=+0.003479921 | 42=-0.001616067 / 123=+0.005892827 / 2026=-0.004336195 |
| A | joint | trained_ball | all | absolute | constant | 0=-0.0005279121 / 1=-0.0002762417 / 2=+0.0002869798 | 42=+0.0004113611 / 123=-0.0003131446 / 2026=-0.000624665 |
| A | joint | trained_ball | all | dual | constant | 0=+0.0002070734 / 1=-0.0004124275 / 2=+0.0003317993 | 42=+0.0002244197 / 123=+4.055551e-05 / 2026=-0.0001342277 |
| A | joint | trained_ball | 1001_plus | absolute | constant | 0=+0.00809332 / 1=-0.004373428 / 2=+0.001412678 | 42=+0.007961046 / 123=+2.28135e-06 / 2026=-0.002830757 |
| A | joint | trained_ball | 1001_plus | dual | constant | 0=+0.0003905644 / 1=-0.004092873 / 2=+0.0003619923 | 42=+0.004515099 / 123=-0.003375815 / 2026=-0.004479601 |
| A | joint | matched_norm | all | absolute | constant | 0=+0.0003120795 / 1=-0.0002631956 / 2=-0.001219775 | 42=+0.0009764355 / 123=-0.0009108688 / 2026=-0.001218135 |
| A | joint | matched_norm | all | dual | constant | 0=+0.0002200031 / 1=-0.0003055178 / 2=-0.0003795737 | 42=+0.0002497526 / 123=-0.0001998133 / 2026=-0.0005052441 |
| A | joint | matched_norm | 1001_plus | absolute | constant | 0=+0.009263033 / 1=-0.0068771 / 2=-0.005330474 | 42=+0.005621087 / 123=-0.007396551 / 2026=-0.001169077 |
| A | joint | matched_norm | 1001_plus | dual | constant | 0=+0.00641302 / 1=-0.004987978 / 2=-0.00516893 | 42=+0.003035012 / 123=-0.003892497 / 2026=-0.002886403 |
| A | interaction | trained_ball | all | absolute | constant | 0=-0.0004720202 / 1=-0.0003952049 / 2=+0.000269823 | 42=+0.0003826034 / 123=-0.0002745808 / 2026=-0.0007125435 |
| A | interaction | trained_ball | all | dual | constant | 0=+0.0002437707 / 1=-0.0005459307 / 2=+0.0002974291 | 42=+0.0001999549 / 123=+4.361973e-05 / 2026=-0.0002419051 |
| A | interaction | trained_ball | 1001_plus | absolute | constant | 0=+0.007127748 / 1=-0.004578206 / 2=+0.002403944 | 42=+0.007865272 / 123=-0.0003526671 / 2026=-0.002559119 |
| A | interaction | trained_ball | 1001_plus | dual | constant | 0=-0.001101342 / 1=-0.003748673 / 2=+0.001303636 | 42=+0.004585787 / 123=-0.004236759 / 2026=-0.003895406 |
| A | interaction | matched_norm | all | absolute | constant | 0=+0.0008426258 / 1=-0.0004005442 / 2=-0.003162428 | 42=+0.001144089 / 123=-0.002395351 / 2026=-0.001423447 |
| A | interaction | matched_norm | all | dual | constant | 0=+0.0003941327 / 1=+0.0002232843 / 2=-0.002434902 | 42=-0.0002732883 / 123=-0.0008733046 / 2026=-0.0006448063 |
| A | interaction | matched_norm | 1001_plus | absolute | constant | 0=+0.008311364 / 1=-0.008208239 / 2=-0.008482963 | 42=+0.005316487 / 123=-0.01401838 / 2026=+0.0003220515 |
| A | interaction | matched_norm | 1001_plus | dual | constant | 0=+0.004967485 / 1=-3.086227e-06 / 2=-0.008648851 | 42=+0.00465108 / 123=-0.009785324 / 2026=+0.001449792 |

## 独立核验与解释范围

全部NPZ监督与冻结源缓存逐值对齐，电芯窗口MAE、RMSE和bias独立重算最大差为1e-16。实际增量CSV覆盖49680条cell×layer记录，范数控制全部通过为True。

所有95%区间采用训练seed与电芯或工况的配对交叉重抽样，抽样次数2000。区间用于描述当前训练设置，各折训练重叠和源历史暴露限制独立确认性解释。每折、每种子、每个工况的反方向结果完整保留。

## 优势与风险

优势为共同起点、折内来源、同组预算及独立数值核验。风险包括历史源暴露、有限远期样本、常数电压统计、模型的不同有效自由度、范数匹配评价的分布变化与LoRA零B起点的Jacobian退化。

## 拓展思考

门控收益与Bridge/Head交互、特征的增量预测信息和子空间外更新收益分别以配对结果解释，单项改善的因果归属还取决于其他对照。

## 你可能想知道

本汇总执行0次训练和0次优化更新，正式目标校准次数为0。执行完整性置信度由来源闭环、逐样本重算和预算审计支持。
