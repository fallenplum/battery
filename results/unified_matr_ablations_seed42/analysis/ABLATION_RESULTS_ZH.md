# M4独立消融seed42正式结果

A1至A4各完成原六个划分，共24项正式fit与49800次优化更新。四个fold0完整重复另8400次，四个单epoch工程smoke另56次，实际总更新58256次。对照M4使用已封存预测。

训练目标沿用trajectory MAE + 0.1 × (端点rate MAE × 1000) + 0.001 × persistence penalty。选权使用validation全轨迹cell-mean MAE的strict first-min，训练固定150epochs。

## 全轨迹与四时距

|范围|模型|指标|单位|cell mean|protocol macro|电芯|协议|观测点|
|---|---|---|---|---|---|---|---|---|
|main_oof|A1_M4_LEGACY_FLOW|horizon_1001_1000000_mae|SOH|0.0707599716|0.0667905622|40|19|10614|
|main_oof|A1_M4_LEGACY_FLOW|horizon_1001_1000000_rmse|SOH|0.0732033932|0.069107455|40|19|10614|
|main_oof|A1_M4_LEGACY_FLOW|horizon_101_200_mae|SOH|0.00249466502|0.00354634602|169|77|16865|
|main_oof|A1_M4_LEGACY_FLOW|horizon_101_200_rmse|SOH|0.00288555088|0.00402998252|169|77|16865|
|main_oof|A1_M4_LEGACY_FLOW|horizon_201_500_mae|SOH|0.00744191185|0.0112738544|168|76|49795|
|main_oof|A1_M4_LEGACY_FLOW|horizon_201_500_rmse|SOH|0.00984446788|0.0149764362|168|76|49795|
|main_oof|A1_M4_LEGACY_FLOW|horizon_501_1000_mae|SOH|0.0403698111|0.0402252348|151|61|46157|
|main_oof|A1_M4_LEGACY_FLOW|horizon_501_1000_rmse|SOH|0.0489842518|0.0456578865|151|61|46157|
|main_oof|A1_M4_LEGACY_FLOW|r200_500_absolute_error|SOH/cycle|5.06270013e-05|7.47436102e-05|151|61|302|
|main_oof|A1_M4_LEGACY_FLOW|r500_1000_absolute_error|SOH/cycle|0.000136845486|0.00012950005|42|19|84|
|main_oof|A1_M4_LEGACY_FLOW|trajectory_mae|SOH|0.0201480695|0.0181740996|169|77|123431|
|main_oof|A1_M4_LEGACY_FLOW|trajectory_rmse|SOH|0.0336507828|0.0287205518|169|77|123431|
|main_oof|A2_M4_NO_B_WRITE|horizon_1001_1000000_mae|SOH|0.082459188|0.08302552|40|19|10614|
|main_oof|A2_M4_NO_B_WRITE|horizon_1001_1000000_rmse|SOH|0.0866206063|0.0864463581|40|19|10614|
|main_oof|A2_M4_NO_B_WRITE|horizon_101_200_mae|SOH|0.00583071959|0.00889268174|169|77|16865|
|main_oof|A2_M4_NO_B_WRITE|horizon_101_200_rmse|SOH|0.00614314509|0.0094145364|169|77|16865|
|main_oof|A2_M4_NO_B_WRITE|horizon_201_500_mae|SOH|0.00919774569|0.0130341237|168|76|49795|
|main_oof|A2_M4_NO_B_WRITE|horizon_201_500_rmse|SOH|0.0122270697|0.0176626956|168|76|49795|
|main_oof|A2_M4_NO_B_WRITE|horizon_501_1000_mae|SOH|0.0454779851|0.0474677602|151|61|46157|
|main_oof|A2_M4_NO_B_WRITE|horizon_501_1000_rmse|SOH|0.054890619|0.0535933339|151|61|46157|
|main_oof|A2_M4_NO_B_WRITE|r200_500_absolute_error|SOH/cycle|7.00969988e-05|9.66994027e-05|151|61|302|
|main_oof|A2_M4_NO_B_WRITE|r500_1000_absolute_error|SOH/cycle|0.000128056001|0.000127117094|42|19|84|
|main_oof|A2_M4_NO_B_WRITE|trajectory_mae|SOH|0.0231130742|0.0210105833|169|77|123431|
|main_oof|A2_M4_NO_B_WRITE|trajectory_rmse|SOH|0.038428373|0.0326108419|169|77|123431|
|main_oof|A3_M4_STATIC_RHO|horizon_1001_1000000_mae|SOH|0.0753312483|0.0746761697|40|19|10614|
|main_oof|A3_M4_STATIC_RHO|horizon_1001_1000000_rmse|SOH|0.0784310569|0.0773628641|40|19|10614|
|main_oof|A3_M4_STATIC_RHO|horizon_101_200_mae|SOH|0.00595514916|0.00878544577|169|77|16865|
|main_oof|A3_M4_STATIC_RHO|horizon_101_200_rmse|SOH|0.00639169582|0.00953090015|169|77|16865|
|main_oof|A3_M4_STATIC_RHO|horizon_201_500_mae|SOH|0.00961314694|0.0142794218|168|76|49795|
|main_oof|A3_M4_STATIC_RHO|horizon_201_500_rmse|SOH|0.012557247|0.0188733187|168|76|49795|
|main_oof|A3_M4_STATIC_RHO|horizon_501_1000_mae|SOH|0.0465595869|0.0485853515|151|61|46157|
|main_oof|A3_M4_STATIC_RHO|horizon_501_1000_rmse|SOH|0.0569069797|0.0551524268|151|61|46157|
|main_oof|A3_M4_STATIC_RHO|r200_500_absolute_error|SOH/cycle|6.60763538e-05|9.42506359e-05|151|61|302|
|main_oof|A3_M4_STATIC_RHO|r500_1000_absolute_error|SOH/cycle|0.00013082636|0.000130964903|42|19|84|
|main_oof|A3_M4_STATIC_RHO|trajectory_mae|SOH|0.0229458599|0.0215712917|169|77|123431|
|main_oof|A3_M4_STATIC_RHO|trajectory_rmse|SOH|0.0384695138|0.0332801674|169|77|123431|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|horizon_1001_1000000_mae|SOH|0.080933376|0.0791058086|40|19|10614|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|horizon_1001_1000000_rmse|SOH|0.0838021556|0.0814632936|40|19|10614|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|horizon_101_200_mae|SOH|0.00614738769|0.00910874895|169|77|16865|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|horizon_101_200_rmse|SOH|0.00660833495|0.0098459295|169|77|16865|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|horizon_201_500_mae|SOH|0.0106502448|0.0154077566|168|76|49795|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|horizon_201_500_rmse|SOH|0.0132160192|0.0192061491|168|76|49795|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|horizon_501_1000_mae|SOH|0.0458298447|0.045744331|151|61|46157|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|horizon_501_1000_rmse|SOH|0.0560532087|0.0522961761|151|61|46157|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|r200_500_absolute_error|SOH/cycle|6.27490236e-05|8.41728328e-05|151|61|302|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|r500_1000_absolute_error|SOH/cycle|0.000142005266|0.000141128071|42|19|84|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|trajectory_mae|SOH|0.0244055925|0.0228891983|169|77|123431|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|trajectory_rmse|SOH|0.0395555066|0.0339689728|169|77|123431|
|main_oof|M4_FULL_METHOD|horizon_1001_1000000_mae|SOH|0.0829638039|0.0856235031|40|19|10614|
|main_oof|M4_FULL_METHOD|horizon_1001_1000000_rmse|SOH|0.0859831765|0.0881833496|40|19|10614|
|main_oof|M4_FULL_METHOD|horizon_101_200_mae|SOH|0.00537210603|0.00792659671|169|77|16865|
|main_oof|M4_FULL_METHOD|horizon_101_200_rmse|SOH|0.00581530902|0.00867831007|169|77|16865|
|main_oof|M4_FULL_METHOD|horizon_201_500_mae|SOH|0.0101244148|0.0146156331|168|76|49795|
|main_oof|M4_FULL_METHOD|horizon_201_500_rmse|SOH|0.0130499844|0.0190123046|168|76|49795|
|main_oof|M4_FULL_METHOD|horizon_501_1000_mae|SOH|0.0450655648|0.0446560865|151|61|46157|
|main_oof|M4_FULL_METHOD|horizon_501_1000_rmse|SOH|0.0552985275|0.0512080189|151|61|46157|
|main_oof|M4_FULL_METHOD|r200_500_absolute_error|SOH/cycle|6.45485993e-05|8.48677461e-05|151|61|302|
|main_oof|M4_FULL_METHOD|r500_1000_absolute_error|SOH/cycle|0.000146507053|0.000152684831|42|19|84|
|main_oof|M4_FULL_METHOD|trajectory_mae|SOH|0.0232710748|0.0215282057|169|77|123431|
|main_oof|M4_FULL_METHOD|trajectory_rmse|SOH|0.0385274196|0.032855653|169|77|123431|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|horizon_1001_1000000_mae|SOH|0.111940705|0.125972519|14|6|6267|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|horizon_1001_1000000_rmse|SOH|0.115180769|0.12928947|14|6|6267|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|horizon_101_200_mae|SOH|0.00351157514|0.00604280809|46|15|4571|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|horizon_101_200_rmse|SOH|0.00415267873|0.00704029653|46|15|4571|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|horizon_201_500_mae|SOH|0.00806430819|0.0127319411|45|14|13378|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|horizon_201_500_rmse|SOH|0.0106565941|0.0174456089|45|14|13378|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|horizon_501_1000_mae|SOH|0.044338109|0.0460672092|40|11|14096|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|horizon_501_1000_rmse|SOH|0.0527680531|0.0521181894|40|11|14096|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|r200_500_absolute_error|SOH/cycle|6.4031557e-05|8.73671946e-05|40|11|80|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|r500_1000_absolute_error|SOH/cycle|0.000152377426|0.000180575586|14|6|28|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|trajectory_mae|SOH|0.0277652302|0.0265467883|46|15|38312|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|trajectory_rmse|SOH|0.0427901481|0.0395434338|46|15|38312|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|horizon_1001_1000000_mae|SOH|0.17501994|0.158458004|14|6|6267|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|horizon_1001_1000000_rmse|SOH|0.189666995|0.17007537|14|6|6267|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|horizon_101_200_mae|SOH|0.0061524526|0.0111479363|46|15|4571|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|horizon_101_200_rmse|SOH|0.00659253369|0.0120732032|46|15|4571|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|horizon_201_500_mae|SOH|0.0119697546|0.0197964639|45|14|13378|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|horizon_201_500_rmse|SOH|0.0158464538|0.0270504489|45|14|13378|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|horizon_501_1000_mae|SOH|0.0487366013|0.0650281905|40|11|14096|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|horizon_501_1000_rmse|SOH|0.0578669207|0.0715621222|40|11|14096|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|r200_500_absolute_error|SOH/cycle|8.16817235e-05|0.000137350373|40|11|80|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|r500_1000_absolute_error|SOH/cycle|0.000156823208|0.000169987616|14|6|28|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|trajectory_mae|SOH|0.0422406696|0.0427779462|46|15|38312|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|trajectory_rmse|SOH|0.0645175173|0.0631728231|46|15|38312|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|horizon_1001_1000000_mae|SOH|0.24526354|0.209224529|14|6|6267|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|horizon_1001_1000000_rmse|SOH|0.273986302|0.229634067|14|6|6267|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|horizon_101_200_mae|SOH|0.0075322624|0.012464004|46|15|4571|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|horizon_101_200_rmse|SOH|0.00780333966|0.013069749|46|15|4571|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|horizon_201_500_mae|SOH|0.0106119318|0.0169037396|45|14|13378|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|horizon_201_500_rmse|SOH|0.0139701417|0.0229459017|45|14|13378|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|horizon_501_1000_mae|SOH|0.0461511384|0.0551513906|40|11|14096|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|horizon_501_1000_rmse|SOH|0.0558576832|0.0621853495|40|11|14096|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|r200_500_absolute_error|SOH/cycle|7.11214661e-05|0.000103292532|40|11|80|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|r500_1000_absolute_error|SOH/cycle|0.000159899347|0.000177188676|14|6|28|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|trajectory_mae|SOH|0.0515979926|0.0474401968|46|15|38312|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|trajectory_rmse|SOH|0.0815597503|0.0717382517|46|15|38312|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|horizon_1001_1000000_mae|SOH|0.19393549|0.177742399|14|6|6267|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|horizon_1001_1000000_rmse|SOH|0.21007905|0.19038839|14|6|6267|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|horizon_101_200_mae|SOH|0.0064297559|0.0125005909|46|15|4571|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|horizon_101_200_rmse|SOH|0.00693849501|0.0136978372|46|15|4571|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|horizon_201_500_mae|SOH|0.00977568566|0.018313347|45|14|13378|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|horizon_201_500_rmse|SOH|0.0124881069|0.0234570131|45|14|13378|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|horizon_501_1000_mae|SOH|0.0413466759|0.0434527892|40|11|14096|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|horizon_501_1000_rmse|SOH|0.0526410817|0.0519186954|40|11|14096|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|r200_500_absolute_error|SOH/cycle|4.5330369e-05|6.45241288e-05|40|11|80|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|r500_1000_absolute_error|SOH/cycle|0.000167684782|0.000184490163|14|6|28|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|trajectory_mae|SOH|0.0424686588|0.0415454765|46|15|38312|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|trajectory_rmse|SOH|0.0660328994|0.0604193265|46|15|38312|
|unseen_protocol_sensitivity|M4_FULL_METHOD|horizon_1001_1000000_mae|SOH|0.20930777|0.183884133|14|6|6267|
|unseen_protocol_sensitivity|M4_FULL_METHOD|horizon_1001_1000000_rmse|SOH|0.230511652|0.200041006|14|6|6267|
|unseen_protocol_sensitivity|M4_FULL_METHOD|horizon_101_200_mae|SOH|0.00598265507|0.0112138224|46|15|4571|
|unseen_protocol_sensitivity|M4_FULL_METHOD|horizon_101_200_rmse|SOH|0.00648281942|0.0122214159|46|15|4571|
|unseen_protocol_sensitivity|M4_FULL_METHOD|horizon_201_500_mae|SOH|0.0140976463|0.0240971982|45|14|13378|
|unseen_protocol_sensitivity|M4_FULL_METHOD|horizon_201_500_rmse|SOH|0.0177151128|0.0306939558|45|14|13378|
|unseen_protocol_sensitivity|M4_FULL_METHOD|horizon_501_1000_mae|SOH|0.0504334467|0.0594644259|40|11|14096|
|unseen_protocol_sensitivity|M4_FULL_METHOD|horizon_501_1000_rmse|SOH|0.0608001824|0.0669442593|40|11|14096|
|unseen_protocol_sensitivity|M4_FULL_METHOD|r200_500_absolute_error|SOH/cycle|7.47844584e-05|0.000111525072|40|11|80|
|unseen_protocol_sensitivity|M4_FULL_METHOD|r500_1000_absolute_error|SOH/cycle|0.000167432375|0.000178408484|14|6|28|
|unseen_protocol_sensitivity|M4_FULL_METHOD|trajectory_mae|SOH|0.0482477825|0.0472276771|46|15|38312|
|unseen_protocol_sensitivity|M4_FULL_METHOD|trajectory_rmse|SOH|0.0747149734|0.0693738856|46|15|38312|

## 训练与选权

|模型|split|epoch|selected|val MAE|末epoch loss|selected epoch loss|updates|
|---|---|---|---|---|---|---|---|
|A1_M4_LEGACY_FLOW|cell_fold_0|150|89|0.022839488|0.0077166663|0.013742246|2100|
|A1_M4_LEGACY_FLOW|cell_fold_1|150|143|0.022634236|0.010475017|0.0092262147|2100|
|A1_M4_LEGACY_FLOW|cell_fold_2|150|114|0.016705704|0.0082623809|0.010017941|2100|
|A1_M4_LEGACY_FLOW|cell_fold_3|150|82|0.018056843|0.0078889217|0.01444813|2100|
|A1_M4_LEGACY_FLOW|cell_fold_4|150|60|0.016735711|0.0089116181|0.01914454|2100|
|A1_M4_LEGACY_FLOW|unseen_protocol_sensitivity|150|81|0.018312635|0.0087795732|0.018527156|1950|
|A2_M4_NO_B_WRITE|cell_fold_0|150|90|0.024469066|0.016691029|0.025264205|2100|
|A2_M4_NO_B_WRITE|cell_fold_1|150|127|0.022895672|0.016740827|0.020001025|2100|
|A2_M4_NO_B_WRITE|cell_fold_2|150|138|0.022523682|0.014561657|0.018808322|2100|
|A2_M4_NO_B_WRITE|cell_fold_3|150|103|0.023232867|0.016623004|0.02260812|2100|
|A2_M4_NO_B_WRITE|cell_fold_4|150|128|0.018936004|0.018296726|0.018757665|2100|
|A2_M4_NO_B_WRITE|unseen_protocol_sensitivity|150|107|0.01708413|0.021027404|0.026959233|1950|
|A3_M4_STATIC_RHO|cell_fold_0|150|105|0.02385721|0.014393252|0.020604197|2100|
|A3_M4_STATIC_RHO|cell_fold_1|150|130|0.022297698|0.017419945|0.01768292|2100|
|A3_M4_STATIC_RHO|cell_fold_2|150|108|0.021497419|0.014340966|0.020733946|2100|
|A3_M4_STATIC_RHO|cell_fold_3|150|59|0.023479778|0.01334257|0.02810968|2100|
|A3_M4_STATIC_RHO|cell_fold_4|150|95|0.018807705|0.016198373|0.025978455|2100|
|A3_M4_STATIC_RHO|unseen_protocol_sensitivity|150|143|0.017117156|0.019063332|0.018978775|1950|
|A4_M4_NO_LOCAL_PHYSICS|cell_fold_0|150|126|0.02152472|0.014721188|0.019228327|2100|
|A4_M4_NO_LOCAL_PHYSICS|cell_fold_1|150|142|0.022004038|0.016288311|0.017891262|2100|
|A4_M4_NO_LOCAL_PHYSICS|cell_fold_2|150|125|0.022563076|0.014494645|0.017721449|2100|
|A4_M4_NO_LOCAL_PHYSICS|cell_fold_3|150|49|0.024193935|0.013542835|0.030277818|2100|
|A4_M4_NO_LOCAL_PHYSICS|cell_fold_4|150|84|0.019186555|0.016992105|0.027050175|2100|
|A4_M4_NO_LOCAL_PHYSICS|unseen_protocol_sensitivity|150|110|0.01860868|0.021022477|0.026174157|1950|

## 全轨迹配对差异

误差差值为消融误差减完整M4误差，正值支持完整M4。区间为条件于已拟合seed42预测的配对bootstrap区间。

|范围|消融|指标|估计量|M4改进|95%区间|配对电芯|协议组|
|---|---|---|---|---|---|---|---|
|main_oof|A1_M4_LEGACY_FLOW|trajectory_mae|cell_mean|-0.0031230053|[-0.0050916994, -0.0010932502]|169|77|
|main_oof|A1_M4_LEGACY_FLOW|trajectory_mae|protocol_macro|-0.0033541061|[-0.0054876577, -0.00098737567]|169|77|
|main_oof|A2_M4_NO_B_WRITE|trajectory_mae|cell_mean|-0.00015800055|[-0.001737426, 0.001533201]|169|77|
|main_oof|A2_M4_NO_B_WRITE|trajectory_mae|protocol_macro|-0.00051762237|[-0.0017021418, 0.00067758278]|169|77|
|main_oof|A3_M4_STATIC_RHO|trajectory_mae|cell_mean|-0.00032521485|[-0.0010451336, 0.0003953609]|169|77|
|main_oof|A3_M4_STATIC_RHO|trajectory_mae|protocol_macro|4.3085966e-05|[-0.00083807578, 0.00091774137]|169|77|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|trajectory_mae|cell_mean|0.0011345177|[-0.00036224937, 0.0026907369]|169|77|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|trajectory_mae|protocol_macro|0.0013609926|[-0.00011400938, 0.0029095301]|169|77|
|main_oof|A1_M4_LEGACY_FLOW|trajectory_rmse|cell_mean|-0.0048766368|[-0.0081759368, -0.0016410565]|169|77|
|main_oof|A1_M4_LEGACY_FLOW|trajectory_rmse|protocol_macro|-0.0041351013|[-0.007448018, -0.00068276983]|169|77|
|main_oof|A2_M4_NO_B_WRITE|trajectory_rmse|cell_mean|-9.9046533e-05|[-0.0027519612, 0.0026987228]|169|77|
|main_oof|A2_M4_NO_B_WRITE|trajectory_rmse|protocol_macro|-0.00024481114|[-0.0019105179, 0.0014928084]|169|77|
|main_oof|A3_M4_STATIC_RHO|trajectory_rmse|cell_mean|-5.7905761e-05|[-0.0011834194, 0.0010763475]|169|77|
|main_oof|A3_M4_STATIC_RHO|trajectory_rmse|protocol_macro|0.00042451433|[-0.000908122, 0.0016776966]|169|77|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|trajectory_rmse|cell_mean|0.0010280871|[-0.0013150429, 0.0035128975]|169|77|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|trajectory_rmse|protocol_macro|0.0011133198|[-0.00098934923, 0.0032038406]|169|77|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|trajectory_mae|cell_mean|-0.020482552|[-0.034794872, -0.0086400668]|46|15|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|trajectory_mae|protocol_macro|-0.020680889|[-0.041184701, -0.0069485206]|46|15|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|trajectory_mae|cell_mean|-0.0060071129|[-0.010499499, -0.0016472651]|46|15|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|trajectory_mae|protocol_macro|-0.0044497309|[-0.010083481, 0.00037069407]|46|15|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|trajectory_mae|cell_mean|0.0033502101|[-0.0027601245, 0.010656766]|46|15|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|trajectory_mae|protocol_macro|0.00021251977|[-0.0090766064, 0.012733186]|46|15|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|trajectory_mae|cell_mean|-0.0057791237|[-0.011696872, -0.00030368911]|46|15|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|trajectory_mae|protocol_macro|-0.0056822006|[-0.014325514, 0.0014900147]|46|15|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|trajectory_rmse|cell_mean|-0.031924825|[-0.05452009, -0.012991577]|46|15|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|trajectory_rmse|protocol_macro|-0.029830452|[-0.063252385, -0.0084200284]|46|15|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|trajectory_rmse|cell_mean|-0.010197456|[-0.017429033, -0.0037343979]|46|15|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|trajectory_rmse|protocol_macro|-0.0062010625|[-0.014129505, 0.0010711399]|46|15|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|trajectory_rmse|cell_mean|0.0068447769|[-0.0023107082, 0.018026529]|46|15|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|trajectory_rmse|protocol_macro|0.0023643661|[-0.009910527, 0.02067048]|46|15|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|trajectory_rmse|cell_mean|-0.008682074|[-0.016882656, -0.0010996984]|46|15|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|trajectory_rmse|protocol_macro|-0.0089545591|[-0.02009891, 0.00046845115]|46|15|

各时距及rate的完整配对区间、逐cell正负差异保存在full_vs_ablation_paired_comparisons.csv和full_vs_ablation_paired_cells.csv。


## EOL80

|范围|模型|Harrell C|可比较对|风险电芯|事件|删失|
|---|---|---|---|---|---|---|
|main_oof|M4_FULL_METHOD|0.82671557|14179|169|169|0|
|main_oof|A1_M4_LEGACY_FLOW|0.85139996|14179|169|169|0|
|main_oof|A2_M4_NO_B_WRITE|0.83729459|14179|169|169|0|
|main_oof|A3_M4_STATIC_RHO|0.82572819|14179|169|169|0|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|0.80957754|14179|169|169|0|
|unseen_protocol_sensitivity|M4_FULL_METHOD|0.80426357|1032|46|46|0|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|0.83817829|1032|46|46|0|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|0.81782946|1032|46|46|0|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|0.81589147|1032|46|46|0|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|0.78875969|1032|46|46|0|

|范围|消融|重采样单位|M4 C改进|95%区间|风险集电芯|
|---|---|---|---|---|---|
|main_oof|A1_M4_LEGACY_FLOW|cell|-0.024684392|[-0.044519113, -0.0048220111]|169|
|main_oof|A2_M4_NO_B_WRITE|cell|-0.010579025|[-0.033705253, 0.012271549]|169|
|main_oof|A3_M4_STATIC_RHO|cell|0.0009873757|[-0.012061034, 0.014399765]|169|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|cell|0.017138021|[-0.00070937205, 0.035435373]|169|
|main_oof|A1_M4_LEGACY_FLOW|protocol_group|-0.024684392|[-0.050677316, -0.0033232683]|169|
|main_oof|A2_M4_NO_B_WRITE|protocol_group|-0.010579025|[-0.040196612, 0.014165994]|169|
|main_oof|A3_M4_STATIC_RHO|protocol_group|0.0009873757|[-0.012145906, 0.015046189]|169|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|protocol_group|0.017138021|[-0.0032641396, 0.03700445]|169|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|cell|-0.033914729|[-0.082529772, 0.00893766]|46|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|cell|-0.013565891|[-0.054247351, 0.022885572]|46|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|cell|-0.011627907|[-0.06138766, 0.032517149]|46|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|cell|0.015503876|[-0.032740538, 0.067671333]|46|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|protocol_group|-0.033914729|[-0.088923248, 0.02789176]|46|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|protocol_group|-0.013565891|[-0.048880284, 0.02489113]|46|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|protocol_group|-0.011627907|[-0.061729006, 0.04785923]|46|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|protocol_group|0.015503876|[-0.021447391, 0.062243302]|46|

## 重复、趋势与查询限制

- A1_M4_LEGACY_FLOW完整fold0重复为PASS，150epoch核心历史（epoch、loss、validation MAE、selected、updates）及最终model/optimizer/RNG/selected与预测共20项严格比较，运行耗时独立记录。
- A2_M4_NO_B_WRITE完整fold0重复为PASS，150epoch核心历史（epoch、loss、validation MAE、selected、updates）及最终model/optimizer/RNG/selected与预测共20项严格比较，运行耗时独立记录。
- A3_M4_STATIC_RHO完整fold0重复为PASS，150epoch核心历史（epoch、loss、validation MAE、selected、updates）及最终model/optimizer/RNG/selected与预测共20项严格比较，运行耗时独立记录。
- A4_M4_NO_LOCAL_PHYSICS完整fold0重复为PASS，150epoch核心历史（epoch、loss、validation MAE、selected、updates）及最终model/optimizer/RNG/selected与预测共20项严格比较，运行耗时独立记录。

A1保留原解析Flow，其signed rate可为负，趋势回升按原预测记录。A2至A4保留非负rate积分。观测残差会影响最终prediction的局部变化。

|范围|模型|实际相邻观测对|raw trend上升|clamped trend上升|prediction上升|
|---|---|---|---|---|---|
|main_oof|A1_M4_LEGACY_FLOW|123262|2484|2484|1399|
|main_oof|A2_M4_NO_B_WRITE|123262|0|0|0|
|main_oof|A3_M4_STATIC_RHO|123262|0|0|0|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|123262|0|0|0|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|38266|0|0|0|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|38266|0|0|0|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|38266|0|0|0|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|38266|0|0|0|

查询诊断分别统计真实observed future点、每cell最后有效随访之后至5000，以及all100..5000。率单位为SOH/1000cycles，除以1000为SOH/cycle。A1的signed loss兼容字段来自解析decoder差值，其signed rate来自该差值导数。

未截断raw趋势和raw加观测残差的越界clip统计保存在query_range_summary.csv。有限性与单调性属于数值性质，超出实际随访的曲线仍缺少物理校准。MATR_b1c2的逐模型完整轨迹误差与clip分母保存在known_counterexample_MATR_b1c2.csv。

|范围|模型|查询区域|查询点|raw低于0比例|观测clip比例|rate最小/1000|rate最大/1000|
|---|---|---|---|---|---|---|---|
|main_oof|A1_M4_LEGACY_FLOW|all_queries_100_5000|828269|0.169791|0.182646|-0.0632427|2.18747|
|main_oof|A1_M4_LEGACY_FLOW|beyond_cell_last_followup|704630|0.199584|0.214694|2.77162e-05|2.1769|
|main_oof|A1_M4_LEGACY_FLOW|observed_future|123431|0|0|-0.0632384|2.18747|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|all_queries_100_5000|225446|0.0132493|0.0225242|5.4061e-05|1.3932|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|beyond_cell_last_followup|187079|0.0159665|0.0271436|5.4061e-05|1.3885|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|observed_future|38312|0|0|0.00740458|1.3932|
|main_oof|A2_M4_NO_B_WRITE|all_queries_100_5000|828269|0.665537|0.66445|0.0050632|6.72821|
|main_oof|A2_M4_NO_B_WRITE|beyond_cell_last_followup|704630|0.782317|0.781038|0.0812289|6.72821|
|main_oof|A2_M4_NO_B_WRITE|observed_future|123431|0|0|0.00509001|1.54096|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|all_queries_100_5000|225446|0.615336|0.614098|0.00788105|3.0178|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|beyond_cell_last_followup|187079|0.741532|0.74004|0.125034|3.0178|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|observed_future|38312|0|0|0.00791211|0.91068|
|main_oof|A3_M4_STATIC_RHO|all_queries_100_5000|828269|0.607398|0.605313|0.00599058|6.11423|
|main_oof|A3_M4_STATIC_RHO|beyond_cell_last_followup|704630|0.713976|0.711525|0.0958467|6.11423|
|main_oof|A3_M4_STATIC_RHO|observed_future|123431|0|0|0.00600529|1.38168|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|all_queries_100_5000|225446|0.722772|0.722741|0.000919976|5.65554|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|beyond_cell_last_followup|187079|0.865068|0.864918|0.0440642|5.65554|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|observed_future|38312|0.0289726|0.0295208|0.000924915|2.1419|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|all_queries_100_5000|828269|0.593935|0.591979|0.00515973|8.54832|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|beyond_cell_last_followup|704630|0.698151|0.695852|0.0622452|8.54832|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|observed_future|123431|0|0|0.00517921|1.43863|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|all_queries_100_5000|225446|0.617092|0.61589|0.0057262|2.92768|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|beyond_cell_last_followup|187079|0.743531|0.742055|0.119666|2.92768|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|observed_future|38312|0.000574233|0.00070474|0.00575014|0.961682|
|main_oof|M4_FULL_METHOD|all_queries_100_5000|828269|0.606691|0.604994|0.00469507|6.22004|
|main_oof|M4_FULL_METHOD|beyond_cell_last_followup|704630|0.713144|0.711151|0.063613|6.22004|
|main_oof|M4_FULL_METHOD|observed_future|123431|0|0|0.00471795|1.33943|
|unseen_protocol_sensitivity|M4_FULL_METHOD|all_queries_100_5000|225446|0.666421|0.665689|0.00289849|4.13869|
|unseen_protocol_sensitivity|M4_FULL_METHOD|beyond_cell_last_followup|187079|0.802137|0.801228|0.136467|4.13869|
|unseen_protocol_sensitivity|M4_FULL_METHOD|observed_future|38312|0.00467217|0.00480267|0.00291086|1.25721|

|MATR_b1c2范围|模型|MAE|RMSE|实际观测点|观测clip点|
|---|---|---|---|---|---|
|main_oof|A1_M4_LEGACY_FLOW|0.012376464|0.015735141|2136|0|
|main_oof|A2_M4_NO_B_WRITE|0.10072985|0.15841624|2136|0|
|main_oof|A3_M4_STATIC_RHO|0.038042476|0.048177641|2136|0|
|main_oof|A4_M4_NO_LOCAL_PHYSICS|0.02454129|0.029079074|2136|0|
|main_oof|M4_FULL_METHOD|0.036140235|0.045334465|2136|0|
|unseen_protocol_sensitivity|A1_M4_LEGACY_FLOW|0.078286818|0.099161165|2136|0|
|unseen_protocol_sensitivity|A2_M4_NO_B_WRITE|0.24641236|0.35302577|2136|0|
|unseen_protocol_sensitivity|A3_M4_STATIC_RHO|0.34690315|0.50334671|2136|496|
|unseen_protocol_sensitivity|A4_M4_NO_LOCAL_PHYSICS|0.25352303|0.36706445|2136|27|
|unseen_protocol_sensitivity|M4_FULL_METHOD|0.27755119|0.41116407|2136|184|

## 比较口径、优势与风险

配对误差改进定义为消融误差减完整M4误差，正值支持完整M4。EOL改进定义为完整M4的C减少融C。保留逐cell、协议组及2000次seed42配对bootstrap。bootstrap条件于已拟合预测。

优势来自共同输入、标签、split、训练与选权预算以及独立的单项机制替换。风险包括单seed方差未覆盖、结构性参数量差异、169电芯与M4历史结果已经暴露，以及A4保留raw/reference VIQ中的局部曲线信息。内部网络写入诊断的电化学因果解释仍需独立测量。
