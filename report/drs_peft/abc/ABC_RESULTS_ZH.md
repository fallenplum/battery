# A–C 源工况诊断实验结果

## 执行范围

本报告覆盖A、B、C。新增真实神经训练99项，复用已完成冻结训练36项，新增优化更新95040次，独立重算270项外层结果与51750条电芯窗口误差。最终目标Query未来标签使用量为0。

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
| B | frozen | relative | matched_norm | 1001_plus | 0.096961 | 21 | 13 |
| B | frozen | relative | matched_norm | all | 0.033008 | 115 | 62 |
| B | frozen | relative | trained_ball | 1001_plus | 0.100669 | 21 | 13 |
| B | frozen | relative | trained_ball | all | 0.025773 | 115 | 62 |
| B | frozen | shuffled_dual | matched_norm | 1001_plus | 0.104007 | 21 | 13 |
| B | frozen | shuffled_dual | matched_norm | all | 0.028323 | 115 | 62 |
| B | frozen | shuffled_dual | trained_ball | 1001_plus | 0.102719 | 21 | 13 |
| B | frozen | shuffled_dual | trained_ball | all | 0.025796 | 115 | 62 |
| B | joint | relative | matched_norm | 1001_plus | 0.091780 | 21 | 13 |
| B | joint | relative | matched_norm | all | 0.025692 | 115 | 62 |
| B | joint | relative | trained_ball | 1001_plus | 0.092721 | 21 | 13 |
| B | joint | relative | trained_ball | all | 0.025335 | 115 | 62 |
| B | joint | shuffled_dual | matched_norm | 1001_plus | 0.092127 | 21 | 13 |
| B | joint | shuffled_dual | matched_norm | all | 0.026314 | 115 | 62 |
| B | joint | shuffled_dual | trained_ball | 1001_plus | 0.091937 | 21 | 13 |
| B | joint | shuffled_dual | trained_ball | all | 0.025440 | 115 | 62 |
| C | frozen | fixed_rank4 | matched_norm | 1001_plus | 0.106572 | 21 | 13 |
| C | frozen | fixed_rank4 | matched_norm | all | 0.033831 | 115 | 62 |
| C | frozen | fixed_rank4 | trained_ball | 1001_plus | 0.101795 | 21 | 13 |
| C | frozen | fixed_rank4 | trained_ball | all | 0.025819 | 115 | 62 |
| C | frozen | matched_fixed | matched_norm | 1001_plus | 0.103638 | 21 | 13 |
| C | frozen | matched_fixed | matched_norm | all | 0.025825 | 115 | 62 |
| C | frozen | matched_fixed | trained_ball | 1001_plus | 0.103569 | 21 | 13 |
| C | frozen | matched_fixed | trained_ball | all | 0.025817 | 115 | 62 |
| C | frozen | matched_lora | matched_norm | 1001_plus | 0.103153 | 21 | 13 |
| C | frozen | matched_lora | matched_norm | all | 0.025718 | 115 | 62 |
| C | frozen | matched_lora | trained_ball | 1001_plus | 0.103153 | 21 | 13 |
| C | frozen | matched_lora | trained_ball | all | 0.025718 | 115 | 62 |
| C | frozen | matched_residual | matched_norm | 1001_plus | 0.102469 | 21 | 13 |
| C | frozen | matched_residual | matched_norm | all | 0.025564 | 115 | 62 |
| C | frozen | matched_residual | trained_ball | 1001_plus | 0.102506 | 21 | 13 |
| C | frozen | matched_residual | trained_ball | all | 0.025562 | 115 | 62 |
| C | frozen | rank4_plus_small | matched_norm | 1001_plus | 0.107756 | 21 | 13 |
| C | frozen | rank4_plus_small | matched_norm | all | 0.026454 | 115 | 62 |
| C | frozen | rank4_plus_small | trained_ball | 1001_plus | 0.103400 | 21 | 13 |
| C | frozen | rank4_plus_small | trained_ball | all | 0.025592 | 115 | 62 |

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

## B 状态特征的信息诊断

神经对照保留absolute、relative、dual与整电芯配对打乱视角，并分别评价冻结和联合Bridge/Head。打乱在fit、inner与outer角色内部独立执行无自配置换，使用前100圈特征。

| 对照 | 组 | 评价 | 窗口 | 候选 | 参照 | MAE收益 | 改善% | 电芯CI | 工况CI | 各折均正 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neural_feature_information | frozen | trained_ball | all | absolute | shuffled_dual | 0.000078 | 0.303595 | [-0.00020408006424179118, 0.00029428639986130113] | [-0.0002280949534184393, 0.00021627459982995152] | False |
| neural_feature_information | frozen | trained_ball | all | relative | shuffled_dual | 0.000023 | 0.089397 | [-0.00023510356614784187, 0.00020115980254019504] | [-0.00024010481441944322, 0.00017097242403881123] | False |
| neural_feature_information | frozen | trained_ball | all | dual | shuffled_dual | 0.000095 | 0.369504 | [-0.00020745866172197853, 0.000300407417585764] | [-0.00019137313257091738, 0.0002987868925737389] | False |
| neural_feature_information | frozen | trained_ball | all | dual | absolute | 0.000017 | 0.066110 | [-6.173160707925223e-05, 9.66941156942803e-05] | [-1.8954109182967343e-05, 0.00015102732048033628] | True |
| neural_feature_information | frozen | trained_ball | all | dual | relative | 0.000072 | 0.280358 | [-3.951048913277892e-05, 0.00018463073578347684] | [9.113974903636752e-07, 0.00016690971133960405] | True |
| neural_feature_information | frozen | trained_ball | 1001_plus | absolute | shuffled_dual | 0.002247 | 2.187046 | [-0.0009153476844787307, 0.007089995639393615] | [-0.0007123800549168714, 0.00653771870392879] | True |
| neural_feature_information | frozen | trained_ball | 1001_plus | relative | shuffled_dual | 0.002050 | 1.995987 | [-0.0009697997452185917, 0.006878378944734736] | [-0.000552980794111549, 0.006306534565114827] | False |
| neural_feature_information | frozen | trained_ball | 1001_plus | dual | shuffled_dual | 0.002256 | 2.195800 | [-0.0008666615193586934, 0.006757450502505398] | [-0.0007789999774524716, 0.006968829793740557] | True |
| neural_feature_information | frozen | trained_ball | 1001_plus | dual | absolute | 0.000009 | 0.008950 | [-0.0007273637728782591, 0.0006680027145720647] | [-0.0005556768556350351, 0.0012210220376497466] | False |
| neural_feature_information | frozen | trained_ball | 1001_plus | dual | relative | 0.000205 | 0.203883 | [-0.0007754545961271673, 0.0012702005498569592] | [-0.0011249545065079478, 0.0018603929909990831] | False |
| neural_feature_information | frozen | matched_norm | all | absolute | shuffled_dual | -0.004698 | -16.585746 | [-0.019161872531412584, 0.004873712237608028] | [-0.017331977965040494, 0.0044687264662965215] | False |
| neural_feature_information | frozen | matched_norm | all | relative | shuffled_dual | -0.004685 | -16.541354 | [-0.01836861882269579, 0.0054540859960840425] | [-0.018053362725900964, 0.0042120821353226175] | False |
| neural_feature_information | frozen | matched_norm | all | dual | shuffled_dual | -0.004760 | -16.804704 | [-0.01937579637476604, 0.0044328960194929615] | [-0.01713459071506935, 0.0038080638351158376] | False |
| neural_feature_information | frozen | matched_norm | all | dual | absolute | -0.000062 | -0.187808 | [-0.0008164605114725858, 0.0007407459642846414] | [-0.0007296182967427975, 0.0004155902260488757] | False |
| neural_feature_information | frozen | matched_norm | all | dual | relative | -0.000075 | -0.225971 | [-0.0010015722893830423, 0.0007554460609612274] | [-0.0007502352824758706, 0.0005264532190261408] | False |
| neural_feature_information | frozen | matched_norm | 1001_plus | absolute | shuffled_dual | 0.008173 | 7.858092 | [-0.010922016533004392, 0.028809284870384196] | [-0.019744972904294993, 0.02839577651019574] | True |
| neural_feature_information | frozen | matched_norm | 1001_plus | relative | shuffled_dual | 0.007046 | 6.774147 | [-0.012515051528004124, 0.027662035234611065] | [-0.021926129382920895, 0.02448864870282272] | True |
| neural_feature_information | frozen | matched_norm | 1001_plus | dual | shuffled_dual | 0.006341 | 6.097073 | [-0.012919559439940533, 0.024943442453117547] | [-0.026371614195594233, 0.027945715240480527] | True |
| neural_feature_information | frozen | matched_norm | 1001_plus | dual | absolute | -0.001832 | -1.911202 | [-0.0062623372853666745, 0.0021554924183792025] | [-0.0068521119927659226, 0.0033755925314660306] | False |
| neural_feature_information | frozen | matched_norm | 1001_plus | dual | relative | -0.000704 | -0.726272 | [-0.005186652545372671, 0.004047334236997844] | [-0.006330738215627264, 0.005848158206813179] | False |
| neural_feature_information | joint | trained_ball | all | absolute | shuffled_dual | 0.000284 | 1.116431 | [-0.0004814944718908644, 0.0010522456710541487] | [-0.0006905482432072042, 0.0010737245221301116] | False |
| neural_feature_information | joint | trained_ball | all | relative | shuffled_dual | 0.000106 | 0.414870 | [-0.000572624166348753, 0.0008576402516309176] | [-0.0005621651110877237, 0.0007663438043174496] | False |
| neural_feature_information | joint | trained_ball | all | dual | shuffled_dual | 0.000503 | 1.977535 | [-0.00028379998220289074, 0.0013159744160749017] | [-0.0004751028826901349, 0.0010806673073530924] | True |
| neural_feature_information | joint | trained_ball | all | dual | absolute | 0.000219 | 0.870825 | [-0.00025911734915394925, 0.0008272415938858367] | [-0.0004430934085190197, 0.0006445985662391206] | False |
| neural_feature_information | joint | trained_ball | all | dual | relative | 0.000398 | 1.569175 | [-8.180201336327903e-05, 0.0008941796561638055] | [-0.00028303570383348775, 0.0006497557200159138] | False |
| neural_feature_information | joint | trained_ball | 1001_plus | absolute | shuffled_dual | 0.003391 | 3.688779 | [-0.003153561661293087, 0.01013684235463498] | [-0.003970787360054939, 0.011950501833611179] | True |
| neural_feature_information | joint | trained_ball | 1001_plus | relative | shuffled_dual | -0.000784 | -0.853191 | [-0.007534220332997672, 0.006875589249111865] | [-0.009533594650336001, 0.007464457814512578] | False |
| neural_feature_information | joint | trained_ball | 1001_plus | dual | shuffled_dual | 0.000567 | 0.616785 | [-0.005783853149125403, 0.008056822485285407] | [-0.0072362064073126925, 0.008061393589638442] | False |
| neural_feature_information | joint | trained_ball | 1001_plus | dual | absolute | -0.002824 | -3.189652 | [-0.008269766830772574, 0.0023661840346638967] | [-0.007883841137604125, 0.001587503752348093] | False |
| neural_feature_information | joint | trained_ball | 1001_plus | dual | relative | 0.001351 | 1.457541 | [-0.002166018913683557, 0.004998787100160416] | [-0.001703302540627224, 0.007740340007728177] | False |
| neural_feature_information | joint | matched_norm | all | absolute | shuffled_dual | 0.000370 | 1.405159 | [-0.0006960686372794125, 0.0012933397964490217] | [-0.0011966646415759537, 0.0014567676870076505] | False |
| neural_feature_information | joint | matched_norm | all | relative | shuffled_dual | 0.000622 | 2.362192 | [-0.0006401701746517933, 0.001943126000266094] | [-0.0007712913078783588, 0.0017470459159049676] | True |
| neural_feature_information | joint | matched_norm | all | dual | shuffled_dual | 0.000602 | 2.288428 | [-0.0004525450174956606, 0.001606887328657865] | [-0.0008078811831540554, 0.0015458410352935006] | False |
| neural_feature_information | joint | matched_norm | all | dual | absolute | 0.000232 | 0.895858 | [-0.0006533367112782375, 0.0010969858978635025] | [-0.0005526517523955374, 0.0008848873959472817] | False |
| neural_feature_information | joint | matched_norm | all | dual | relative | -0.000019 | -0.075549 | [-0.0008124623936623399, 0.0007906221082124137] | [-0.0008069769546045302, 0.000672590838954343] | False |
| neural_feature_information | joint | matched_norm | 1001_plus | absolute | shuffled_dual | 0.000437 | 0.474273 | [-0.008188750153376511, 0.009553386799267266] | [-0.008290937151268905, 0.012480661387710328] | False |
| neural_feature_information | joint | matched_norm | 1001_plus | relative | shuffled_dual | 0.000348 | 0.377545 | [-0.007609087891701414, 0.009823569659643264] | [-0.00857087362301025, 0.01096357730646402] | False |
| neural_feature_information | joint | matched_norm | 1001_plus | dual | shuffled_dual | 0.000170 | 0.185055 | [-0.008514485334097842, 0.009890013838060813] | [-0.006777950770785383, 0.011726200204648525] | False |
| neural_feature_information | joint | matched_norm | 1001_plus | dual | absolute | -0.000266 | -0.290596 | [-0.00523968731129924, 0.005112167203626953] | [-0.004208181709964293, 0.00449703486236158] | False |
| neural_feature_information | joint | matched_norm | 1001_plus | dual | relative | -0.000177 | -0.193219 | [-0.006730974135116801, 0.00616293899542369] | [-0.005631149200280579, 0.00984002877806176] | False |

### 退化速率的简单模型诊断

Ridge固定alpha为1。GBR固定100棵树、深度2、最小叶样本数3、学习率0.05。所有组使用相同的六维早期容量参照，附加视角统一为24列。零填充产生的有效秩差异、Ridge有效自由度和实际树叶数保留于各训练凭据。

平均退化速率由窗首尾各20圈的实测SOH均值差除以对应实测循环均值差计算。每窗要求至少80%实测覆盖，首尾各至少10个实测点。

200–500圈支持104颗电芯、51个工况组，各fold fit为57、54、55颗，inner为14、14、14颗。500–1000圈支持23颗电芯、13个工况组，各fold fit为13、13、11颗，inner为3、3、3颗。

| window | model | arm | MAE | physical_cells | protocol_groups |
| --- | --- | --- | --- | --- | --- |
| 200_500 | gbr | absolute | 0.000119 | 104 | 51 |
| 200_500 | gbr | baseline | 0.000113 | 104 | 51 |
| 200_500 | gbr | dual | 0.000126 | 104 | 51 |
| 200_500 | gbr | relative | 0.000116 | 104 | 51 |
| 200_500 | gbr | shuffled_dual | 0.000123 | 104 | 51 |
| 200_500 | ridge | absolute | 0.000130 | 104 | 51 |
| 200_500 | ridge | baseline | 0.000141 | 104 | 51 |
| 200_500 | ridge | dual | 0.000134 | 104 | 51 |
| 200_500 | ridge | relative | 0.000147 | 104 | 51 |
| 200_500 | ridge | shuffled_dual | 0.000167 | 104 | 51 |
| 500_1000 | gbr | absolute | 0.000284 | 23 | 13 |
| 500_1000 | gbr | baseline | 0.000232 | 23 | 13 |
| 500_1000 | gbr | dual | 0.000299 | 23 | 13 |
| 500_1000 | gbr | relative | 0.000279 | 23 | 13 |
| 500_1000 | gbr | shuffled_dual | 0.000273 | 23 | 13 |
| 500_1000 | ridge | absolute | 0.000280 | 23 | 13 |
| 500_1000 | ridge | baseline | 0.000244 | 23 | 13 |
| 500_1000 | ridge | dual | 0.000196 | 23 | 13 |
| 500_1000 | ridge | relative | 0.000203 | 23 | 13 |
| 500_1000 | ridge | shuffled_dual | 0.000421 | 23 | 13 |

| 窗口 | 模型 | 候选 | 参照 | 参照MAE | 候选MAE | 改善% | 各折均正 | 电芯CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200_500 | ridge | absolute | baseline | 0.000141 | 0.000130 | 8.257846 | True | [2.731257552799256e-07, 2.2185749841350533e-05] |
| 200_500 | ridge | absolute | shuffled_dual | 0.000167 | 0.000130 | 22.285402 | True | [6.831542990435693e-06, 9.63813683927425e-05] |
| 200_500 | ridge | relative | baseline | 0.000141 | 0.000147 | -3.940005 | False | [-1.3938334514621125e-05, 2.5240592528906616e-06] |
| 200_500 | ridge | relative | shuffled_dual | 0.000167 | 0.000147 | 11.952627 | False | [-8.374962896560115e-06, 7.75290384592679e-05] |
| 200_500 | ridge | dual | baseline | 0.000141 | 0.000134 | 5.029397 | False | [-6.2050430876237695e-06, 1.965725770238162e-05] |
| 200_500 | ridge | dual | shuffled_dual | 0.000167 | 0.000134 | 19.550589 | True | [1.4701427392873922e-06, 9.572761512478198e-05] |
| 200_500 | ridge | dual | absolute | 0.000130 | 0.000134 | -3.519046 | False | [-1.2287651948221894e-05, 3.3173830918892473e-06] |
| 200_500 | ridge | dual | relative | 0.000147 | 0.000134 | 8.629403 | True | [-1.3574364794173352e-06, 2.5573472436883147e-05] |
| 200_500 | gbr | absolute | baseline | 0.000113 | 0.000119 | -4.911732 | False | [-1.5695533092641694e-05, 5.078149915750438e-06] |
| 200_500 | gbr | absolute | shuffled_dual | 0.000123 | 0.000119 | 3.362905 | False | [-7.753331690398873e-06, 1.6422157679081163e-05] |
| 200_500 | gbr | relative | baseline | 0.000113 | 0.000116 | -2.398361 | False | [-1.500703240374059e-05, 9.686703435601751e-06] |
| 200_500 | gbr | relative | shuffled_dual | 0.000123 | 0.000116 | 5.678040 | False | [-6.8505826722872246e-06, 2.089925553241661e-05] |
| 200_500 | gbr | dual | baseline | 0.000113 | 0.000126 | -11.154156 | False | [-2.5177456699975624e-05, -2.6824750553599126e-07] |
| 200_500 | gbr | dual | shuffled_dual | 0.000123 | 0.000126 | -2.387164 | False | [-1.5212494028024094e-05, 8.832149544467468e-06] |
| 200_500 | gbr | dual | absolute | 0.000119 | 0.000126 | -5.950167 | False | [-1.3192802133218244e-05, -1.5887516261269188e-06] |
| 200_500 | gbr | dual | relative | 0.000116 | 0.000126 | -8.550718 | False | [-1.9976519436083057e-05, 9.841121959592867e-07] |
| 500_1000 | ridge | absolute | baseline | 0.000244 | 0.000280 | -14.623265 | False | [-7.082754954523663e-05, 7.392353375673995e-07] |
| 500_1000 | ridge | absolute | shuffled_dual | 0.000421 | 0.000280 | 33.546054 | True | [1.9368688554201227e-05, 0.00032100069769938557] |
| 500_1000 | ridge | relative | baseline | 0.000244 | 0.000203 | 16.918914 | True | [-7.402729264479715e-06, 8.795633497244337e-05] |
| 500_1000 | ridge | relative | shuffled_dual | 0.000421 | 0.000203 | 51.832938 | True | [8.570305850685484e-05, 0.00041938850043233] |
| 500_1000 | ridge | dual | baseline | 0.000244 | 0.000196 | 19.522438 | False | [-1.1884535805519678e-05, 0.00010318567292662772] |
| 500_1000 | ridge | dual | shuffled_dual | 0.000421 | 0.000196 | 53.342356 | True | [9.563716208120412e-05, 0.000434190606308426] |
| 500_1000 | ridge | dual | absolute | 0.000280 | 0.000196 | 29.789505 | True | [4.0403254112021626e-05, 0.00012990494967017369] |
| 500_1000 | ridge | dual | relative | 0.000203 | 0.000196 | 3.133714 | False | [-2.170375802414752e-05, 3.558924741056311e-05] |
| 500_1000 | gbr | absolute | baseline | 0.000232 | 0.000284 | -22.415623 | False | [-0.00010153864527799604, 1.7821170691148552e-06] |
| 500_1000 | gbr | absolute | shuffled_dual | 0.000273 | 0.000284 | -4.210915 | False | [-8.338771467251242e-05, 5.3102523221549475e-05] |
| 500_1000 | gbr | relative | baseline | 0.000232 | 0.000279 | -20.031005 | False | [-0.00012270488451747067, 2.288591217756824e-05] |
| 500_1000 | gbr | relative | shuffled_dual | 0.000273 | 0.000279 | -2.180920 | False | [-7.113359029463346e-05, 5.403692158768974e-05] |
| 500_1000 | gbr | dual | baseline | 0.000232 | 0.000299 | -29.034754 | False | [-0.00013905238750559952, -6.734547585444014e-07] |
| 500_1000 | gbr | dual | shuffled_dual | 0.000273 | 0.000299 | -9.845700 | False | [-9.460565727951374e-05, 3.3689566564618506e-05] |
| 500_1000 | gbr | dual | absolute | 0.000284 | 0.000299 | -5.407097 | False | [-7.620184866860583e-05, 3.625992363364905e-05] |
| 500_1000 | gbr | dual | relative | 0.000279 | 0.000299 | -7.501186 | False | [-4.7336688401657e-05, 2.865072149339636e-06] |

### 退化速率对照的逐折与逐种子方向

| 实验 | 模型或组 | 评价 | 窗口 | 候选 | 参照 | 各折MAE收益 | 各种子MAE收益 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B_simple | ridge | rate | 200_500 | absolute | baseline | 0=+3.234425e-09 / 1=+2.167312e-05 / 2=+1.238839e-05 | 42=+1.167243e-05 / 123=+1.167243e-05 / 2026=+1.167243e-05 |
| B_simple | ridge | rate | 200_500 | absolute | shuffled_dual | 0=+8.065469e-05 / 1=+2.854631e-05 / 2=+5.088261e-06 | 42=+2.594132e-05 / 123=+5.868108e-05 / 2026=+2.693608e-05 |
| B_simple | ridge | rate | 200_500 | relative | baseline | 0=-1.24369e-05 / 1=-5.797319e-06 / 2=+1.140759e-06 | 42=-5.569181e-06 / 123=-5.569181e-06 / 2026=-5.569181e-06 |
| B_simple | ridge | rate | 200_500 | relative | shuffled_dual | 0=+6.821455e-05 / 1=+1.075866e-06 / 2=-6.15937e-06 | 42=+8.699712e-06 / 123=+4.143947e-05 / 2026=+9.694472e-06 |
| B_simple | ridge | rate | 200_500 | dual | baseline | 0=-1.849397e-06 / 1=+1.307227e-05 / 2=+9.421932e-06 | 42=+7.109032e-06 / 123=+7.109032e-06 / 2026=+7.109032e-06 |
| B_simple | ridge | rate | 200_500 | dual | shuffled_dual | 0=+7.880205e-05 / 1=+1.994546e-05 / 2=+2.121804e-06 | 42=+2.137792e-05 / 123=+5.411768e-05 / 2026=+2.237268e-05 |
| B_simple | ridge | rate | 200_500 | dual | absolute | 0=-1.852632e-06 / 1=-8.600852e-06 / 2=-2.966457e-06 | 42=-4.563399e-06 / 123=-4.563399e-06 / 2026=-4.563399e-06 |
| B_simple | ridge | rate | 200_500 | dual | relative | 0=+1.058751e-05 / 1=+1.886959e-05 / 2=+8.281173e-06 | 42=+1.267821e-05 / 123=+1.267821e-05 / 2026=+1.267821e-05 |
| B_simple | gbr | rate | 200_500 | absolute | baseline | 0=-1.433451e-05 / 1=+3.687861e-06 / 2=-6.826601e-06 | 42=-6.208564e-06 / 123=-4.863371e-06 / 2026=-5.635958e-06 |
| B_simple | gbr | rate | 200_500 | absolute | shuffled_dual | 0=-1.713042e-05 / 1=+3.135963e-05 / 2=-3.803497e-06 | 42=+2.926501e-06 / 123=+1.399095e-06 / 2026=+8.093263e-06 |
| B_simple | gbr | rate | 200_500 | relative | baseline | 0=-3.315364e-06 / 1=-1.846456e-05 / 2=+1.40374e-05 | 42=-2.972509e-06 / 123=-2.26589e-06 / 2026=-2.919936e-06 |
| B_simple | gbr | rate | 200_500 | relative | shuffled_dual | 0=-6.11127e-06 / 1=+9.207206e-06 / 2=+1.70605e-05 | 42=+6.162555e-06 / 123=+3.996575e-06 / 2026=+1.080929e-05 |
| B_simple | gbr | rate | 200_500 | dual | baseline | 0=-1.2347e-05 / 1=-1.98946e-05 / 2=-5.476472e-06 | 42=-1.363097e-05 / 123=-1.190816e-05 / 2026=-1.240317e-05 |
| B_simple | gbr | rate | 200_500 | dual | shuffled_dual | 0=-1.51429e-05 / 1=+7.777165e-06 / 2=-2.453368e-06 | 42=-4.495903e-06 / 123=-5.645695e-06 / 2026=+1.326047e-06 |
| B_simple | gbr | rate | 200_500 | dual | absolute | 0=+1.987514e-06 / 1=-2.358246e-05 / 2=+1.350129e-06 | 42=-7.422404e-06 / 123=-7.04479e-06 / 2026=-6.767217e-06 |
| B_simple | gbr | rate | 200_500 | dual | relative | 0=-9.031634e-06 / 1=-1.430041e-06 / 2=-1.951387e-05 | 42=-1.065846e-05 / 123=-9.64227e-06 / 2026=-9.483239e-06 |
| B_simple | ridge | rate | 500_1000 | absolute | baseline | 0=-4.914578e-05 / 1=-4.692758e-05 / 2=-1.6452e-05 | 42=-3.567746e-05 / 123=-3.567746e-05 / 2026=-3.567746e-05 |
| B_simple | ridge | rate | 500_1000 | absolute | shuffled_dual | 0=+0.0003464788 / 1=+3.545613e-05 / 2=+6.370778e-05 | 42=+3.868672e-05 / 123=+0.000266886 / 2026=+0.0001179378 |
| B_simple | ridge | rate | 500_1000 | relative | baseline | 0=+0.0001143266 / 1=+7.245283e-06 / 2=+1.093311e-05 | 42=+4.127832e-05 / 123=+4.127832e-05 / 2026=+4.127832e-05 |
| B_simple | ridge | rate | 500_1000 | relative | shuffled_dual | 0=+0.0005099512 / 1=+8.962899e-05 / 2=+9.109289e-05 | 42=+0.0001156425 / 123=+0.0003438418 / 2026=+0.0001948936 |
| B_simple | ridge | rate | 500_1000 | dual | baseline | 0=+0.0001297263 / 1=-3.331188e-06 / 2=+2.341461e-05 | 42=+4.763033e-05 / 123=+4.763033e-05 / 2026=+4.763033e-05 |
| B_simple | ridge | rate | 500_1000 | dual | shuffled_dual | 0=+0.0005253509 / 1=+7.905252e-05 / 2=+0.0001035744 | 42=+0.0001219945 / 123=+0.0003501938 / 2026=+0.0002012456 |
| B_simple | ridge | rate | 500_1000 | dual | absolute | 0=+0.0001788721 / 1=+4.359639e-05 / 2=+3.98666e-05 | 42=+8.330778e-05 / 123=+8.330778e-05 / 2026=+8.330778e-05 |
| B_simple | ridge | rate | 500_1000 | dual | relative | 0=+1.539972e-05 / 1=-1.057647e-05 / 2=+1.248149e-05 | 42=+6.352009e-06 / 123=+6.352009e-06 / 2026=+6.352009e-06 |
| B_simple | gbr | rate | 500_1000 | absolute | baseline | 0=+1.809338e-05 / 1=-8.92977e-05 / 2=-7.755855e-05 | 42=-4.385096e-05 / 123=-6.013539e-05 / 2026=-5.207329e-05 |
| B_simple | gbr | rate | 500_1000 | absolute | shuffled_dual | 0=+8.634666e-06 / 1=-9.457904e-06 / 2=-2.869591e-05 | 42=+2.400998e-05 / 123=-1.794255e-05 / 2026=-4.050559e-05 |
| B_simple | gbr | rate | 500_1000 | relative | baseline | 0=-2.803848e-06 / 1=-8.849926e-05 / 2=-4.778378e-05 | 42=-4.463069e-05 / 123=-5.64739e-05 / 2026=-3.835313e-05 |
| B_simple | gbr | rate | 500_1000 | relative | shuffled_dual | 0=-1.226256e-05 / 1=-8.659467e-06 / 2=+1.078862e-06 | 42=+2.323026e-05 / 123=-1.428106e-05 / 2026=-2.678543e-05 |
| B_simple | gbr | rate | 500_1000 | dual | baseline | 0=-5.29753e-06 / 1=-0.0001215384 / 2=-7.354539e-05 | 42=-6.040635e-05 / 123=-7.292624e-05 / 2026=-6.881004e-05 |
| B_simple | gbr | rate | 500_1000 | dual | shuffled_dual | 0=-1.475624e-05 / 1=-4.169864e-05 / 2=-2.468275e-05 | 42=+7.454589e-06 / 123=-3.07334e-05 / 2026=-5.724235e-05 |
| B_simple | gbr | rate | 500_1000 | dual | absolute | 0=-2.339091e-05 / 1=-3.224074e-05 / 2=+4.013165e-06 | 42=-1.655539e-05 / 123=-1.279085e-05 / 2026=-1.673676e-05 |
| B_simple | gbr | rate | 500_1000 | dual | relative | 0=-2.493682e-06 / 1=-3.303917e-05 / 2=-2.576161e-05 | 42=-1.577567e-05 / 123=-1.645235e-05 / 2026=-3.045692e-05 |

## C 固定子空间外的更新

三组主比较各训练9204个有效参数，分别为扩展固定子空间、原rank4加双侧正交残差，以及去除尺度冗余的rank1 BA。另保留192参数原rank4与300参数小残差补充的次级嵌套对照。Bridge/Head与C0全部冻结，新增方向只由本折fit梯度构造。

匹配容量比较涉及方向分配变化。192与300参数比较包含容量差。残差适配覆盖双侧正交块，完整G减去P_U G P_V还包含单侧交叉块，独立梯度诊断分别保留这些部分。

| 对照 | 组 | 评价 | 窗口 | 候选 | 参照 | MAE收益 | 改善% | 电芯CI | 工况CI | 各折均正 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capacity_different_secondary | frozen | trained_ball | all | rank4_plus_small | fixed_rank4 | 0.000227 | 0.879849 | [2.6803672743180554e-05, 0.0005934700019711415] | [3.5460374682220386e-05, 0.0005306403878977364] | True |
| matched_capacity | frozen | trained_ball | all | matched_residual | matched_fixed | 0.000255 | 0.987391 | [3.761997771423088e-05, 0.0004913663671877238] | [-8.56208876462419e-06, 0.0003663612936987144] | True |
| matched_capacity | frozen | trained_ball | all | matched_lora | matched_fixed | 0.000099 | 0.382333 | [-8.550705517633296e-05, 0.0003333626333115169] | [-0.0001652231552458381, 0.00019117212980485594] | False |
| matched_capacity | frozen | trained_ball | all | matched_residual | matched_lora | 0.000156 | 0.607380 | [-2.7241343921129787e-05, 0.00033362816007674476] | [-2.6852976006999255e-06, 0.0003264724707114155] | True |
| capacity_different_secondary | frozen | trained_ball | all | matched_fixed | fixed_rank4 | 0.000002 | 0.008807 | [-0.00024797616747341265, 0.0003037429367040114] | [-0.0001374127268643527, 0.0003207275627728882] | False |
| capacity_different_secondary | frozen | trained_ball | 1001_plus | rank4_plus_small | fixed_rank4 | -0.001605 | -1.576944 | [-0.005413617459093659, 0.00039106828295098117] | [-0.0067560287465325865, 0.0007795110889881843] | False |
| matched_capacity | frozen | trained_ball | 1001_plus | matched_residual | matched_fixed | 0.001062 | 1.025550 | [-0.0012452686629678235, 0.003474512787438793] | [-0.0006235470763504725, 0.004744077748003469] | True |
| matched_capacity | frozen | trained_ball | 1001_plus | matched_lora | matched_fixed | 0.000416 | 0.401619 | [-0.0016351446867933887, 0.0026137443187128017] | [-0.0017101901810058348, 0.0036820925662645735] | False |
| matched_capacity | frozen | trained_ball | 1001_plus | matched_residual | matched_lora | 0.000646 | 0.626447 | [-0.0014521369577067166, 0.0028063368288684374] | [-0.0016788600366794632, 0.0036728956460010116] | False |
| capacity_different_secondary | frozen | trained_ball | 1001_plus | matched_fixed | fixed_rank4 | -0.001774 | -1.742683 | [-0.006102933749140611, 0.0019118136321551597] | [-0.007642862682624013, 0.0020310215813840837] | False |
| capacity_different_secondary | frozen | matched_norm | all | rank4_plus_small | fixed_rank4 | 0.007377 | 21.805025 | [0.00014707385047364423, 0.020537262050893262] | [0.00011430338878939138, 0.01767543315690332] | True |
| matched_capacity | frozen | matched_norm | all | matched_residual | matched_fixed | 0.000260 | 1.007957 | [5.101712107779525e-05, 0.0004889584192975297] | [7.720838494938901e-06, 0.0003695428354228335] | True |
| matched_capacity | frozen | matched_norm | all | matched_lora | matched_fixed | 0.000106 | 0.412321 | [-8.031632888263422e-05, 0.0003279104271986404] | [-0.00014240505446149279, 0.00018737024415722272] | False |
| matched_capacity | frozen | matched_norm | all | matched_residual | matched_lora | 0.000154 | 0.598101 | [-3.5763262813363096e-05, 0.0003392469610812513] | [-2.5578527886851217e-06, 0.00032975208841975253] | True |
| capacity_different_secondary | frozen | matched_norm | all | matched_fixed | fixed_rank4 | 0.008007 | 23.666240 | [1.022818034546768e-05, 0.01898271028467153] | [-8.207403126069328e-06, 0.01803134446582714] | True |
| capacity_different_secondary | frozen | matched_norm | 1001_plus | rank4_plus_small | fixed_rank4 | -0.001184 | -1.110948 | [-0.02516776817087833, 0.01642853181459075] | [-0.023557765150067817, 0.018902067106570518] | False |
| matched_capacity | frozen | matched_norm | 1001_plus | matched_residual | matched_fixed | 0.001169 | 1.128426 | [-0.001338399267871055, 0.0034700831853375675] | [-0.0004978927003395164, 0.004899212917515642] | True |
| matched_capacity | frozen | matched_norm | 1001_plus | matched_lora | matched_fixed | 0.000486 | 0.468480 | [-0.0016906468516364424, 0.002805808839004736] | [-0.0016830150907072112, 0.004045014500120341] | False |
| matched_capacity | frozen | matched_norm | 1001_plus | matched_residual | matched_lora | 0.000684 | 0.663052 | [-0.0012889510239653735, 0.00297932406426743] | [-0.0015273448571303514, 0.003696680383825233] | False |
| capacity_different_secondary | frozen | matched_norm | 1001_plus | matched_fixed | fixed_rank4 | 0.002934 | 2.753256 | [-0.020343562111069503, 0.0216415814936726] | [-0.01992874287593874, 0.02860322941082985] | False |

### 独立工况梯度的能量分解

下表先在每个工况内平均，再对工况与训练seed等权平均。G_perp包含双侧正交和两种单侧交叉块，保留其相对能量与训练梯度余弦。

| layer | fixed_energy_ratio | left_only_energy_ratio | right_only_energy_ratio | double_perp_energy_ratio | residual_energy_ratio | residual_to_fit_warmup_residual_cosine | full_to_fit_warmup_mean_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| encoder.layers.0.self_attn.q_proj | 0.924982 | 0.000431 | 0.074188 | 0.000399 | 0.075018 | 0.055688 | 0.132444 |
| encoder.layers.0.self_attn.v_proj | 0.863843 | 0.000579 | 0.134955 | 0.000623 | 0.136157 | 0.017130 | 0.127376 |
| encoder.layers.1.self_attn.q_proj | 0.652991 | 0.009492 | 0.298655 | 0.038861 | 0.347009 | 0.047516 | 0.107980 |
| encoder.layers.1.self_attn.v_proj | 0.835226 | 0.005206 | 0.156858 | 0.002710 | 0.164774 | 0.029619 | 0.129327 |
| encoder.layers.2.self_attn.q_proj | 0.692585 | 0.022533 | 0.237048 | 0.047835 | 0.307415 | 0.057000 | 0.105728 |
| encoder.layers.2.self_attn.v_proj | 0.831510 | 0.010247 | 0.149554 | 0.008689 | 0.168490 | 0.050195 | 0.125307 |
| encoder.layers.3.self_attn.q_proj | 0.701882 | 0.045200 | 0.181979 | 0.070939 | 0.298118 | 0.055061 | 0.111520 |
| encoder.layers.3.self_attn.v_proj | 0.776608 | 0.035227 | 0.161179 | 0.026985 | 0.223392 | 0.047123 | 0.127751 |
| encoder.layers.4.self_attn.q_proj | 0.508919 | 0.049796 | 0.283927 | 0.157358 | 0.491081 | 0.064757 | 0.102949 |
| encoder.layers.4.self_attn.v_proj | 0.819208 | 0.045846 | 0.103205 | 0.031741 | 0.180792 | 0.057095 | 0.131380 |
| encoder.layers.5.self_attn.q_proj | 0.648244 | 0.048084 | 0.224098 | 0.079574 | 0.351756 | 0.052852 | 0.113853 |
| encoder.layers.5.self_attn.v_proj | 0.898096 | 0.049608 | 0.032125 | 0.020171 | 0.101904 | 0.068101 | 0.138899 |

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
| B | frozen | trained_ball | all | absolute | shuffled_dual | 0=-0.0002579512 / 1=+0.0002539834 / 2=+0.0002477631 | 42=-7.953194e-05 / 123=+0.0001379599 / 2026=+0.0001765182 |
| B | frozen | trained_ball | all | relative | shuffled_dual | 0=-0.0002708063 / 1=+0.0001813775 / 2=+0.0001663449 | 42=-0.0001273806 / 123=+0.000149972 / 2026=+4.659145e-05 |
| B | frozen | trained_ball | all | dual | shuffled_dual | 0=-0.0002387566 / 1=+0.0002685234 / 2=+0.0002649765 | 42=-8.382481e-05 / 123=+0.0001734595 / 2026=+0.0001963172 |
| B | frozen | trained_ball | all | dual | absolute | 0=+1.919455e-05 / 1=+1.454002e-05 / 2=+1.721339e-05 | 42=-4.292868e-06 / 123=+3.549959e-05 / 2026=+1.979894e-05 |
| B | frozen | trained_ball | all | dual | relative | 0=+3.204964e-05 / 1=+8.714591e-05 / 2=+9.863157e-05 | 42=+4.355582e-05 / 123=+2.348749e-05 / 2026=+0.0001497257 |
| B | frozen | trained_ball | 1001_plus | absolute | shuffled_dual | 0=+0.004964467 / 1=+0.0008771111 / 2=+0.0008979862 | 42=+0.006149286 / 123=-0.0007102164 / 2026=+0.001300495 |
| B | frozen | trained_ball | 1001_plus | relative | shuffled_dual | 0=+0.004944271 / 1=-0.0002128105 / 2=+0.00141934 | 42=+0.005728508 / 123=-0.0008820024 / 2026=+0.001304295 |
| B | frozen | trained_ball | 1001_plus | dual | shuffled_dual | 0=+0.005490802 / 1=+0.0003281321 / 2=+0.0009476082 | 42=+0.005982824 / 123=-0.0002042212 / 2026=+0.0009879393 |
| B | frozen | trained_ball | 1001_plus | dual | absolute | 0=+0.0005263348 / 1=-0.000548979 / 2=+4.962196e-05 | 42=-0.0001664617 / 123=+0.0005059952 / 2026=-0.0003125557 |
| B | frozen | trained_ball | 1001_plus | dual | relative | 0=+0.0005465315 / 1=+0.0005409426 / 2=-0.0004717323 | 42=+0.0002543165 / 123=+0.0006777812 / 2026=-0.0003163559 |
| B | frozen | matched_norm | all | absolute | shuffled_dual | 0=-0.0199385 / 1=+0.001999893 / 2=+0.004246847 | 42=-0.02020803 / 123=+0.005371803 / 2026=+0.0007433954 |
| B | frozen | matched_norm | all | relative | shuffled_dual | 0=-0.019329 / 1=+0.001577119 / 2=+0.004082133 | 42=-0.01995981 / 123=+0.005491787 / 2026=+0.0004129057 |
| B | frozen | matched_norm | all | dual | shuffled_dual | 0=-0.01958208 / 1=+0.001333743 / 2=+0.004359523 | 42=-0.01951734 / 123=+0.004560812 / 2026=+0.0006776455 |
| B | frozen | matched_norm | all | dual | absolute | 0=+0.0003564168 / 1=-0.0006661507 / 2=+0.0001126755 | 42=+0.0006906943 / 123=-0.0008109915 / 2026=-6.574989e-05 |
| B | frozen | matched_norm | all | dual | relative | 0=-0.0002530825 / 1=-0.0002433765 / 2=+0.0002773896 | 42=+0.0004424687 / 123=-0.0009309751 / 2026=+0.0002647398 |
| B | frozen | matched_norm | 1001_plus | absolute | shuffled_dual | 0=+0.006114994 / 1=+0.007317693 / 2=+0.01108615 | 42=+0.00827433 / 123=+0.01237131 / 2026=+0.003873188 |
| B | frozen | matched_norm | 1001_plus | relative | shuffled_dual | 0=+0.007775757 / 1=+0.001024549 / 2=+0.0123364 | 42=+0.005771673 / 123=+0.01194793 / 2026=+0.003417099 |
| B | frozen | matched_norm | 1001_plus | dual | shuffled_dual | 0=+0.00660886 / 1=+0.001001663 / 2=+0.01141358 | 42=+0.006353663 / 123=+0.01164231 / 2026=+0.001028121 |
| B | frozen | matched_norm | 1001_plus | dual | absolute | 0=+0.000493866 / 1=-0.006316031 / 2=+0.0003274321 | 42=-0.001920667 / 123=-0.0007289993 / 2026=-0.002845067 |
| B | frozen | matched_norm | 1001_plus | dual | relative | 0=-0.001166897 / 1=-2.288592e-05 / 2=-0.0009228226 | 42=+0.0005819904 / 123=-0.0003056186 / 2026=-0.002388977 |
| B | joint | trained_ball | all | absolute | shuffled_dual | 0=-0.0004212172 / 1=+0.001123174 / 2=+0.000168665 | 42=+0.0001580473 / 123=+0.0004050427 / 2026=+0.0002889732 |
| B | joint | trained_ball | all | relative | shuffled_dual | 0=-0.0005316992 / 1=+0.001026016 / 2=-0.000160918 | 42=-0.0004782156 / 123=+0.0003000241 / 2026=+0.0004948209 |
| B | joint | trained_ball | all | dual | shuffled_dual | 0=+0.0003137683 / 1=+0.0009869885 / 2=+0.0002134845 | 42=-2.889407e-05 / 123=+0.0007587428 / 2026=+0.0007794105 |
| B | joint | trained_ball | all | dual | absolute | 0=+0.0007349854 / 1=-0.0001361858 / 2=+4.481952e-05 | 42=-0.0001869414 / 123=+0.0003537001 / 2026=+0.0004904373 |
| B | joint | trained_ball | all | dual | relative | 0=+0.0008454674 / 1=-3.902759e-05 / 2=+0.0003744025 | 42=+0.0004493215 / 123=+0.0004587187 / 2026=+0.0002845895 |
| B | joint | trained_ball | 1001_plus | absolute | shuffled_dual | 0=+0.005387048 / 1=+0.003665451 / 2=+0.001121547 | 42=+6.033884e-05 / 123=+0.002938918 / 2026=+0.00717479 |
| B | joint | trained_ball | 1001_plus | relative | shuffled_dual | 0=-0.007871609 / 1=+0.005264241 / 2=+0.0002541754 | 42=-0.005030879 / 123=-0.0002114414 / 2026=+0.002889128 |
| B | joint | trained_ball | 1001_plus | dual | shuffled_dual | 0=-0.002315707 / 1=+0.003946006 / 2=+7.086148e-05 | 42=-0.003385608 / 123=-0.0004391785 / 2026=+0.005525947 |
| B | joint | trained_ball | 1001_plus | dual | absolute | 0=-0.007702755 / 1=+0.0002805545 / 2=-0.001050686 | 42=-0.003445946 / 123=-0.003378097 / 2026=-0.001648843 |
| B | joint | trained_ball | 1001_plus | dual | relative | 0=+0.005555902 / 1=-0.001318236 / 2=-0.0001833139 | 42=+0.001645272 / 123=-0.0002277371 / 2026=+0.002636818 |
| B | joint | matched_norm | all | absolute | shuffled_dual | 0=+0.0007484573 / 1=+0.001524848 / 2=-0.001174022 | 42=+0.0006607961 / 123=+0.0007079557 / 2026=-0.0002595025 |
| B | joint | matched_norm | all | relative | shuffled_dual | 0=+2.369924e-05 / 1=+0.001398239 / 2=+0.0004585388 | 42=-0.0006156008 / 123=+0.00197359 / 2026=+0.000506754 |
| B | joint | matched_norm | all | dual | shuffled_dual | 0=+0.000656381 / 1=+0.001482526 / 2=-0.0003338206 | 42=-6.588677e-05 / 123=+0.001419011 / 2026=+0.0004533886 |
| B | joint | matched_norm | all | dual | absolute | 0=-9.207634e-05 / 1=-4.232214e-05 / 2=+0.0008402015 | 42=-0.0007266829 / 123=+0.0007110554 / 2026=+0.0007128911 |
| B | joint | matched_norm | all | dual | relative | 0=+0.0006326817 / 1=+8.428711e-05 / 2=-0.0007923594 | 42=+0.000549714 / 123=-0.0005545789 / 2026=-5.336547e-05 |
| B | joint | matched_norm | 1001_plus | absolute | shuffled_dual | 0=+0.004140848 / 1=+0.003909078 / 2=-0.006739119 | 42=-0.004233944 / 123=-0.001364646 / 2026=+0.006909397 |
| B | joint | matched_norm | 1001_plus | relative | shuffled_dual | 0=-0.00747786 / 1=+0.007977732 / 2=+0.0005435942 | 42=-0.005486506 / 123=+0.004503606 / 2026=+0.002026367 |
| B | joint | matched_norm | 1001_plus | dual | shuffled_dual | 0=+0.001290835 / 1=+0.0057982 / 2=-0.006577575 | 42=-0.006820019 / 123=+0.002139408 / 2026=+0.005192071 |
| B | joint | matched_norm | 1001_plus | dual | absolute | 0=-0.002850013 / 1=+0.001889122 / 2=+0.0001615442 | 42=-0.002586074 / 123=+0.003504054 / 2026=-0.001717326 |
| B | joint | matched_norm | 1001_plus | dual | relative | 0=+0.008768694 / 1=-0.002179532 / 2=-0.007121169 | 42=-0.001333513 / 123=-0.002364198 / 2026=+0.003165704 |
| C | frozen | trained_ball | all | rank4_plus_small | fixed_rank4 | 0=+0.0005305412 / 1=+6.759732e-05 / 2=+7.538305e-05 | 42=+0.0005281575 / 123=+0.0001204954 / 2026=+3.285217e-05 |
| C | frozen | trained_ball | all | matched_residual | matched_fixed | 0=+0.0005969458 / 1=+9.9699e-05 / 2=+5.909065e-05 | 42=+0.000299416 / 123=+0.000275068 / 2026=+0.0001902524 |
| C | frozen | trained_ball | all | matched_lora | matched_fixed | 0=+0.0003014665 / 1=-3.775137e-06 / 2=-6.909432e-06 | 42=+0.0001873916 / 123=+0.0001194334 / 2026=-1.070726e-05 |
| C | frozen | trained_ball | all | matched_residual | matched_lora | 0=+0.0002954794 / 1=+0.0001034741 / 2=+6.600008e-05 | 42=+0.0001120245 / 123=+0.0001556346 / 2026=+0.0002009597 |
| C | frozen | trained_ball | all | matched_fixed | fixed_rank4 | 0=-9.057189e-05 / 1=+8.268179e-05 / 2=+1.715518e-05 | 42=+0.0001982842 / 123=-2.171061e-05 / 2026=-0.0001697518 |
| C | frozen | trained_ball | 1001_plus | rank4_plus_small | fixed_rank4 | 0=-0.005980183 / 1=+0.0006481002 / 2=+0.0005163487 | 42=-0.004951528 / 123=-0.0001808669 / 2026=+0.0003166606 |
| C | frozen | trained_ball | 1001_plus | matched_residual | matched_fixed | 0=+0.002508176 / 1=+0.0001759117 / 2=+0.0005023566 | 42=+0.002403299 / 123=-0.0003267722 / 2026=+0.001109917 |
| C | frozen | trained_ball | 1001_plus | matched_lora | matched_fixed | 0=+0.001380485 / 1=+0.0003753235 / 2=-0.0005079555 | 42=+0.0003631601 / 123=-0.0006727149 / 2026=+0.001557408 |
| C | frozen | trained_ball | 1001_plus | matched_residual | matched_lora | 0=+0.00112769 / 1=-0.0001994117 / 2=+0.001010312 | 42=+0.002040139 / 123=+0.0003459427 / 2026=-0.0004474909 |
| C | frozen | trained_ball | 1001_plus | matched_fixed | fixed_rank4 | 0=-0.005169372 / 1=-0.000256093 / 2=+0.0001035917 | 42=-0.005403316 / 123=+0.001610643 / 2026=-0.001529199 |
| C | frozen | matched_norm | all | rank4_plus_small | fixed_rank4 | 0=+0.01945193 / 1=+0.0005656063 / 2=+0.001795309 | 42=+0.02011318 / 123=+0.001878021 / 2026=+0.0001394023 |
| C | frozen | matched_norm | all | matched_residual | matched_fixed | 0=+0.0005902288 / 1=+9.9699e-05 / 2=+8.228957e-05 | 42=+0.000299091 / 123=+0.0002682341 / 2026=+0.0002135746 |
| C | frozen | matched_norm | all | matched_lora | matched_fixed | 0=+0.0003014665 / 1=-3.775137e-06 / 2=+1.661731e-05 | 42=+0.0001873916 / 123=+0.0001194334 / 2026=+1.26149e-05 |
| C | frozen | matched_norm | all | matched_residual | matched_lora | 0=+0.0002887623 / 1=+0.0001034741 / 2=+6.567226e-05 | 42=+0.0001116995 / 123=+0.0001488007 / 2026=+0.0002009597 |
| C | frozen | matched_norm | all | matched_fixed | fixed_rank4 | 0=+0.01891447 / 1=+0.0004855251 / 2=+0.004332563 | 42=+0.01982365 / 123=+0.004252451 / 2026=-5.648616e-05 |
| C | frozen | matched_norm | 1001_plus | rank4_plus_small | fixed_rank4 | 0=-0.01299685 / 1=+0.003792454 / 2=+0.005652503 | 42=-0.008346928 / 123=+0.004361867 / 2026=+0.0004331709 |
| C | frozen | matched_norm | 1001_plus | matched_residual | matched_fixed | 0=+0.002623945 / 1=+0.0001759117 / 2=+0.0007085834 | 42=+0.002400806 / 123=-0.0002110027 / 2026=+0.001318637 |
| C | frozen | matched_norm | 1001_plus | matched_lora | matched_fixed | 0=+0.001380485 / 1=+0.0003753235 / 2=-0.0002992355 | 42=+0.0003631601 / 123=-0.0006727149 / 2026=+0.001766128 |
| C | frozen | matched_norm | 1001_plus | matched_residual | matched_lora | 0=+0.00124346 / 1=-0.0001994117 / 2=+0.001007819 | 42=+0.002037646 / 123=+0.0004617122 / 2026=-0.0004474909 |
| C | frozen | matched_norm | 1001_plus | matched_fixed | fixed_rank4 | 0=-0.005583791 / 1=+0.002620456 / 2=+0.01176597 | 42=-0.002960644 / 123=+0.01284887 / 2026=-0.001085597 |

## 独立核验与解释范围

全部NPZ监督与冻结源缓存逐值对齐，电芯窗口MAE、RMSE和bias独立重算最大差为1e-16。实际增量CSV覆盖124200条cell×layer记录，范数控制全部通过为True。

所有95%区间采用训练seed与电芯或工况的配对交叉重抽样，抽样次数2000。区间用于描述当前训练设置，各折训练重叠和源历史暴露限制独立确认性解释。每折、每种子、每个工况的反方向结果完整保留。

## 优势与风险

优势为共同起点、折内来源、同组预算及独立数值核验。风险包括历史源暴露、有限远期样本、常数电压统计、模型的不同有效自由度、范数匹配评价的分布变化与LoRA零B起点的Jacobian退化。

## 拓展思考

门控收益与Bridge/Head交互、特征的增量预测信息和子空间外更新收益分别以配对结果解释，单项改善的因果归属还取决于其他对照。

## 你可能想知道

本汇总执行0次训练和0次优化更新，正式目标校准次数为0。执行完整性置信度由来源闭环、逐样本重算和预算审计支持。
