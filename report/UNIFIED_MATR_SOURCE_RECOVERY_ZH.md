# MATR 统一合同的原始来源恢复与身份审计

## 范围与交付状态

固定主队列为169颗电芯，对应174个跨批仪器测试段。已发布的四份MATLAB档案包含185段，制造条码归一化后对应180颗物理电芯。官方EDP目录共186条仪器测试记录，逐文件HTTP范围请求确认总量为28,061,961,088字节。

四份完整MATLAB档案已恢复，共10,870,637,553字节，逐份完整SHA256及官方URL进入来源注册。固定队列所需174份原始CSV已全部完整恢复，实际总量21,837,210,616字节，全部具有逐文件完整SHA256。来源恢复独立QA的13项检查全部通过。

来源恢复与硬件身份进入本报告。真实循环映射、测量阶段、前缀与后缀隔离以及最终数据冻结资格由统一数据QA记录。

## 官方来源与文件完整性

| 批次 | 官方文件标识 | 完整MATLAB字节数 | 完整SHA256 |
|---|---|---|---|
| b1，2017-05-12 | 5c86c0b5fa2ede00015ddf66 | 3,025,320,241 | 9d928ab978f0e3c70b31cb833a749fedd35094d01af76475d69b40aa3497f5ba |
| b2，2017-06-30 | 5c86bf13fa2ede00015ddd82 | 2,007,331,155 | 63ab200d09ecb237fee5ef3a5c5db76e3212e3206a0bd92f769e1427fed338b8 |
| b3，2018-04-12 | 5c86bd64fa2ede00015ddbb2 | 3,236,690,412 | 62c30e413b63e6144720e016deed3661fac8468641794a5807b123fe84717998 |
| b4，2019-01-24 | 5dcef152110002c7215b2c90 | 2,601,295,745 | 9604dfd58926a99d8aece76e1c407defb7359190edcc21e4d4dd217a726882af |

官方目录提供逐电芯原始CSV以及其所属批次、通道、制造条码和预先指定的实验方案。来源查找实际读取了公开前端资源，确认项目与测试目录位于EDP接口。两项目ID分别为5c48dd2bc625d700019f3204和5d80e633f405260001c0b60a。[官方项目目录](https://data.matr.io/1/api/v1/edp/projects)

前三批完整MATLAB文件还逐段重放了历史原策略恢复时保存的字节范围，全部与历史范围SHA一致。官方文件API没有提供可用的SHA512值，因此本地完整SHA256与HTTP实体ETag分别记录其证据范围。

46份b4原始CSV的官方文件元数据长度与实际资产长度不同。逐份HTTP Content-Range确认后，186份CSV真实总量比元数据总量少58,686,963字节。下载采用实际长度、明确字节区间与对应ETag，完整文件再计算SHA256。

## 硬件身份绑定

MATLAB的barcode与channel_id采用MCOS string存储。h5py直接读出的uint32数组属于对象引用编码。恢复程序解析MCOS类别表、对象表、命名保存属性和UTF16载荷，取得真实条码与通道。

185个MATLAB来源段均在相应官方批次内找到唯一的条码与通道联合匹配。条码归一化规则为去除首尾空白并转为大写ASCII，原值SHA和归一值SHA均保留。固定169颗电芯恰好具有169个不同归一制造条码。

五组续测的条码完全相同，父段与续测段分别为b1c0→b2c7、b1c1→b2c8、b1c2→b2c9、b1c3→b2c15、b1c4→b2c16。来源绑定保留统一cell_uid、独立source_segment_id以及仪器文件身份，分割时可据此保持物理电芯互斥。[原作者续测连接说明](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/Load%20Data.ipynb)

官方EDP额外记录为b4通道46，其CSV文件标识为5dcef3c5110002c7215b2ced，实际大小4,005,938,440字节。该记录位于185段MATLAB快照及本轮固定169队列之外。其余11个固定队列外MATLAB电芯及历史排除原因保留在完整来源表中。

174份原始Arbin Metadata.csv也已全部恢复，共103,450字节，每份保留官方文件URL、完整SHA256、原文件名与全部原字段。共有24种字段，前两批各22列，后两批各24列，包含源表索引、device_id、test_id、iv_ch_id、首次启动、恢复、结束时间、schedule文件名与版本、数据库来源、制造条码、辅助通道标记及CAN/SMB配置占位字段。

设备标识60369369对应前两批89段，设备标识03603603对应后两批85段。设备标识按原字符串保留前导零。174组(device_id, test_id, iv_ch_id)均唯一，原生零起始iv_ch_id加1后全部精确对应EDP与MATLAB的通道。原Metadata.csv的item_id也全部与已认证制造条码一致，独立元数据QA的13项检查通过。

原schedule_file_name有83个不同值，schedule_version有2个不同值。resumed_times与last_resume_datetime在全部174段中均为0，文件内实测的b1c18循环重启仍需从原生行序认证。CAN配置文件名及MD5字段全空，SMB配置字段仅后两批存在且全空。原始空字符串、原列缺失与数值零分别保留。制造条码、数据库字符串和CAN配置字符串的原文保持私有，公开行以SHA及支持计数连接。

元数据完整表为results/unified_matr/instrument_metadata_audit.csv与instrument_metadata_audit.json。各段的开始、恢复、结束时间及数据库信息用于来源审计，新增模型输入列为0。

原Metadata时间与CSV DateTime直接数值比较时存在明显差异。125段的last_end_datetime早于CSV最后DateTime，其中44段差值超过60秒，7段超过3,600秒。CSV首行DateTime减metadata首次启动值的范围为0.0120至98,688.9205秒。跨字段时区与导出语义尚未认证，原始值与数值差进入来源QA，真实轨迹终点由原始CSV行序与阶段记录确定。

## 原始控制协议的全字段恢复

固定队列174个来源段对应174个官方Arbin .sdu控制协议文件标识，全部正文已完整恢复，共3,695,390字节，具有80种不同完整内容SHA。逐段的真实协议文件名与原Metadata.csv记录的schedule_file_name去目录后全部一致。174份协议的声明Step数分布为13步91段、14步4段和15步79段。

所有Schedule、Step、Limit、Equation、Goto、原始单位字段和显示因子均按原字符串保留，空字符串保留其原值。全部80种内容共67,391个内容SHA×section×field组合。原始机器、用户名与路径信息保持私有，公开字段以来源SHA连接。独立SDU元数据QA的13项检查全部通过。

| SDU主放电定义 | 来源段数 | 原控制值 | 原扩展控制值1 |
|---|---|---|---|
| sourceStep10，Discharge，CCCV | 127 | −4.4 | 2 |
| sourceStep12，Discharge，CCCV | 43 | −4.4 | 2 |
| sourceStep9，Discharge，CCCV | 4 | −4.4 | 2 |

sourceStep9对应b3c7、b3c10、b3c16与b3c45，协议名为20180412-4_8C_80per_4_8C_newstructure.sdu。全部原Limit方程与原单位字段可在完整协议表中追溯。初始放电、第二次放电、脉冲放电和最终低电流放电保留各自的label与控制类型，实际原生Step_Index到SDU步骤的连接由逐行测量QA认证。

协议工件为results/unified_matr/schedule_metadata_audit.json、schedule_static_fields.parquet与schedule_metadata_QA.json。恢复与审计新增模型输入列为0，已登记测量阈值变更为0。

## 单位与循环坐标的证据

原作者解析代码已固定到提交aa54406731665e3ef1f77ad43dbc55ff7c763fd3，完整本地副本为6,869字节，SHA256为6cf481b35e05465feb9a5d66381e92cdc934867b173dd51a627184b2bc1d1ce9。代码第91–92行定义时间与电流变换，第154行重新生成summary循环序列。[固定版本原作者代码](https://github.com/chueh-ermon/BMS-autoanalysis/blob/aa54406731665e3ef1f77ad43dbc55ff7c763fd3/cell_analysis.m)

```matlab
battery.cycles(k).I = Current(cycle_indices)/1.1;
battery.summary.cycle = (1:k)';
```

MATLAB的I字段需乘1.1恢复安培，t字段以分钟存储。原CSV保留原生Current、Test_Time、DateTime、Step_Index和Cycle_Index。CSV读取器直接保留其原生电流与时间数值。

MATLAB的summary.cycle由作者生成连续序列。其与逐圈Qd相符可验证档案内部对应关系，仪器真实Cycle_Index需要从原CSV恢复。原始CSV保留初始0圈、终止圈、仪器时间错误及停机信息；完整性与可用性须逐圈区分。

续测CSV会从Cycle_Index=0重新开始。五组边界同时保存原生首末Cycle_Index、Data_Point和DateTime，结合充放电阶段检验接续关系。统一数据处理应保留原生段与本地循环身份，并明确物理全局坐标的来源依据。

同一CSV内也发现仪器计数器重置。b1c18在原生数据行964481→964482处由Cycle_Index=691转为0，Step_Index由11转为0，Data_Point继续递增且DateTime正向推进。完整1,116,181行的独立转移审计复现该边界，并记录6次Test_Time逆序。真实循环身份还需包含文件内native run，逐行的循环下降、正向缺号、Data_Point非递增与时钟逆序进入全174段审计。

原作者源码按全文件唯一Cycle_Index取同号行。该机制结合已实测的文件内计数器重置，会产生跨run同号行合并风险，早期输入可能因此混入后期记录。统一处理采用原始CSV行序、显式native run与原控制器循环标识建立可追溯输入边界。

## 全量原生通道与数值审计

174份CSV共143,396,900条原生数据行。前两批89份文件各15列，后两批85份文件各16列，全部2,695个文件×字段组合均已扫描。每个字段保留总数、有限值数、缺失数、非有限值数、零值数、有限值最小值、最大值和均值。新增的完整通道审计覆盖原生Temperature、Internal_Resistance、Step_Time、dV/dt与Aux_Voltage。

| 原生字段 | 有限值数 | 缺失数 | 最小值 | 最大值 | 解释范围 |
|---|---|---|---|---|---|
| Current | 143,396,900 | 0 | −13.434734 | 9.2818794 | A，原始仪器导出值 |
| Voltage | 143,396,900 | 0 | 0.73577738 | 6.6061416 | V，极值源于b1c18 |
| Charge_Capacity | 143,396,900 | 0 | 0 | 2.9658952 | Ah，极值源于b1c18 |
| Discharge_Capacity | 143,396,900 | 0 | 0 | 2.8840845 | Ah，极值源于b1c18 |
| Temperature | 143,396,856 | 44 | 20.614586 | 44.709492 | °C，44个缺失值保留 |
| Internal_Resistance | 111,506,853 | 31,890,047 | 0 | 0.032501802 | 原字段单位Ω，9,791个零值保留，测量时机与物理R资格待认证 |
| Aux_Voltage | 0 | 62,913,939 | 空 | 空 | 85份含此列的文件中全部缺失 |
| Test_Time | 143,396,900 | 0 | −17,140.3059 | 4,933,650.3498 | s，负值与逆序保留 |
| Step_Time | 143,396,900 | 0 | −2,577,002.7117 | 382,008.6461 | s，负值保留 |

极端值进入全原始记录审计，正式测量的阶段完整性、仪器时钟资格与量纲闭合由逐圈数据处理记录。上述源字段统计保留异常原值，物理R特征仍按其独立测量资格处理。

| 原生行序转移类型 | 次数 |
|---|---|
| Cycle_Index变更 | 140,402 |
| Cycle_Index下降 | 1，仅b1c18 |
| Cycle_Index正向跳号大于1 | 34 |
| Data_Point非递增 | 0 |
| DateTime逆序 | 6 |
| Test_Time逆序 | 500 |

DateTime逆序涉及b1c35、b3c30、b3c40、b1c28、b1c29和b1c4。每种异常均在私有缓存保留前后行及原字段上下文，公开工件保留行号、计数和必要控制器标识。全部原字段扫描与转移扫描均核对完整来源SHA，独立工件QA的16项检查通过。数值扫描先完成的81份文件为新增转移任务补扫，其余文件同一读取流完成两类审计，实际扫描安排进入汇总。

全部185个MATLAB来源段的static字段也已独立核验。185组Vdlin均含1,000个有限单调点，范围3.5→2.0V。官方release的cycle_life在183段为有限值，b3c23和b3c32为非有限值，两者均位于固定169队列之外。此字段仅作来源标签对照。

完整工件为results/unified_matr/native_channel_audit.parquet、native_channel_audit_summary.json、native_channel_audit_QA.json、source_static_field_audit.json和source_recovery_qa.json。公开工件保留去标识统计与完整来源摘要，原生逐行数据、制造条码和第三方源码全文保存在私有目录。

## 重建数据的真实协议步骤复核

最终v4的169电芯、174来源段共140,576条原生记录，包含140,348条regular记录、228条非regular记录。140,295条regular容量有效，53条regular容量无效，全部非regular记录的容量有效掩码为false。最终整表与169个完成缓存的所有列逐行比较完全一致。

174个来源协议均有唯一label恰为Discharge、控制类型CCCV、控制值−4.4且扩展控制值1为2的原始主放电定义。将其步骤索引与140,295条有效regular记录比较，133,515条精确相等，6,780条存在原Step_Index标签例外，涉及111颗电芯，其中386条位于前K100。例外对为expected10→observed9共6,724条、expected12→observed11共56条；observed索引在原协议中的label为Finishing Step共199条、rest3共6,581条。该精确标签相等检查为false，完整记录定位进入SOURCE_DISCHARGE_STEP_QA.json。

全部140,295条有效regular记录的已存字段独立复核均满足既定的零点、电流幅度、起末电压和容量范围条件。Q零点范围0至0.000046693058Ah，CC电流中位数范围−4.4003043至−4.3997431A，完整负相位最小电流范围−4.6828351至−4.4002767A，末端电压范围1.9971452至2.0687876V。复核沿用已经登记的3.3V起点下限、2.1V末端上限和电流噪声容差，新增资格门槛为0。

进一步直接读取了b1c0/native583、b2c10/native394和b4c0/native14的原始CSV。三条记录在原协议主放电索引下的实际行数均为0，约−4.4A至2.0V的完整负相位保留在前一Finishing或rest3索引下。原波形重算的Q零点、CC电流中位数、完整相位电流范围、电压起末与容量均精确等于v4存储值，超过已登记正电流噪声与负激励界限的相位点均为0。三例原CSV重放进入SOURCE_DISCHARGE_STEP_NATIVE_REPLAY.json，完整波形片段保持私有。

原Step标签滞留是得到三例波形支持的解释，控制系统具体原因仍待来源证据。全部6,780条标签例外均已核对存储的计量字段，直接独立重放原CSV的范围为上述三例。物理阶段实际身份与来源标签精确相等分别保留证据与适用范围，本审计修改资格、标签或模型输入的次数为0。

总体来源阶段QA状态为COMPLETE_WITH_SOURCE_STEP_LABEL_EXCEPTIONS。最终数据冻结采用原始I/V/Q与原CCCV(−4.4A，2V)定义共同认证实际主放电，Step标签差异以显式warning保留。原Step标签精确相等为false，8项已经登记的电学与计量要求均实际通过，三例原始CSV重放的全部对照量精确相等。

全部174份原CSV的完整文件SHA、官方文件标识及下载URL进入instrument_bindings.json，对应资产URL与HTTP实体ETag保存在完整下载收据。原生全通道读取流已逐文件独立复核完整SHA。阶段QA锚定该bindings与完整SHA验证工件的摘要，同时记录169个完成缓存、最终全表、配置、重建recipe与审计代码的SHA。三例原始CSV重放各自记录完整来源SHA及私有片段SHA。阶段QA脚本读取全量已存计量字段与三例原波形，整份原CSV的全量SHA复核范围由native_channel_audit_QA.json明确记录，可据同一官方来源和原生行边界重新执行。

## 优势与劣势（风险）

优势为四批完整波形来源可访问、所有固定队列来源身份唯一匹配、原生仪器循环和阶段字段可恢复。完整文件SHA与原始字段支持逐记录追溯，前缀和未来轨迹可以在同一来源定义下建立。

劣势（风险）包括原始Test_Time与Step_Time重置、批次停机、初始和终止记录完整性差异，以及跨批计数器重置。固定169队列继承历史来源筛选与多轮研究暴露。上述因素需要进入资格掩码、分母及结论范围。

## 置信度评级

文件字节数、官方身份联合匹配、MCOS解码结果和源码单位变换的置信度为高，依据为实际HTTP响应、完整文件摘要和185段逐项联合匹配。每颗电芯的真实K100与完整未来轨迹资格以独立仪器映射和统一数据QA的结果为准。

## 拓展思考

169颗固定主队列、174个来源段、185个MATLAB段与186条官方仪器测试分别承担不同覆盖计数，来源表保留这些分母及其连接关系。

## 你可能想知道

官方文件元数据中的size存在陈旧值。实际下载范围、当前HTTP实体及完整文件SHA共同确定本轮恢复快照，后续数据合同可直接引用这些身份。
