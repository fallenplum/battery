# MATR 统一总合同与 seed42 数据工程审查

## 范围与当前状态

本审查逐节读取《SRSR_Physics_统一研究问题_一次性数据处理与模型实验总合同.md》，记录其 SHA256 为 `c59764b219f629eb5767706809a5393536fe977fa6092ae127156340c62a50d7`。用户明确仅用 seed42，正式训练矩阵据此登记 `[42]`。主合同限定 MATR169 LFP/石墨用于主建模与性能评估，HUST 保留仪器语义审计角色，Stanford 历史负结果和 SDU 来源记录继续保存。

本次审查完成合同机器清单、MATR 旧预处理与新 raw_reader 静态核对，以及独立物理特征模块的数学自检。随后独立读取作者导出源码，确认公开 MATLAB 档案的 `summary.cycle` 经过顺序重编号。继续检索官方 EDP 接口后获得原始 CSV 来源和真实 `Cycle_Index`，当前已独立验证完整 b2c0、b1c0 仪器日志与 MATLAB 波形绑定。**DATA-FREEZE 仍待完整主队列来源、续测轴与全部数据重建 QA**，原始仪器日志的实际恢复正在推进。下载进度和单电芯支持数以本报告对应产物为准。

## DATA-FREEZE 必须完整覆盖的内容

| 内容 | 可核验产物与约束 |
|---|---|
| 来源与身份 | 四个完整原始档案与 SHA，`cell_uid`、原始/处理循环、批次、源 segment/record、summary.cycle、测量阶段、协议 UID 和 supplied 映射均可追溯 |
| 放电容量 | 完整且电流符号一致的放电阶段，counter 零点、I–t–Q 单位与积分检查，重置/坏圈/诊断/形成/重复原因逐字段保留 |
| SOH | 同时保存标称容量与早期初始容量归一化，主口径预先登记，未来使用所有有效观测，EOL80 记录首次实际越阈值及右删失 |
| 曲线 | 固定 2.8–3.3 V 的 64 点，电流/连续点/单调性阈值预先登记，无外插，绝对 counter 锚点与 3.3 V 置零形态版本同时保存 |
| 因果隔离 | 全部模型输入来自真实循环 1–100，参考只用真实循环 1–10，后缀扰动/截断不能改变前缀曲线与特征，人口统计量仅训练折拟合 |
| 全部物理特征 | a/b、原始 64 点残差、8 个等宽局部 bin、局部方向/斜率/变化率/支持/噪声与 K10/K50/K100 锚点，值与掩码同维 |
| 划分与暴露 | 唯一物理电芯互斥，跨批续测/生成重复共享身份，精确协议分层与预先固定的未见协议敏感性，169 颗历史暴露逐项公开 |
| 冻结 | manifest、cells、early curves、features、future SOH、split、QA 报告和全局 fingerprint 一次完整产出，正式模型引用同一 fingerprint |

机器可读字段与 18 项断言位于 `configs/unified_matr_seed42_requirements.json`。其中英文 assertion 是执行时的核验要求，清单文件本身不代表对应数据已经认证。不足 1000 圈的电芯仍保留主任务资格，第二段速率单列支持分母。测量不合格电芯留在 169 颗资格表内并记录原因。

## MATR 真实轴与续测风险

读取的旧脚本为 `BatteryLife/process_scripts/preprocess_MATR.py`。`load_batch` 同时读取原始 summary.cycle 和 waveform object 索引，但 supplied `cycle_number` 最终直接使用数组索引。原始 object 0 被删除。b1c0 自索引 11 起、b1c18 自 39 起、b2c12 自 252 起、b2c44 自 247 起改读下一个 waveform，IR 仍按旧 summary 索引取值。这些操作要求保留 summary 行与 waveform 的独立绑定证据。

续测共有五对，旧脚本注释的“四个”与实际数组长度不一致。

| 后续批段 | 唯一物理电芯 |
|---|---|
| b2c7 | b1c0 |
| b2c8 | b1c1 |
| b2c9 | b1c2 |
| b2c15 | b1c3 |
| b2c16 | b1c4 |

旧脚本以第一段 summary 长度平移第二段 summary.cycle，并按 waveform 数组长度拼接。前一版新 `raw_reader.py` 独立保留源段和数组位置，避免复用 supplied 编号，但仍把 summary 的顺序序列作为候选物理轴。作者源码提供了该假设失败的直接证据。修正后的来源语义必须保留 `source_batch/source_cell_index/source_segment_id/source_record_index` 与作者 summary 序列，`original_cycle_id` 继续缺失，`raw_cycle_certified=False`。

作者 `cell_analysis.m` 第27行从原始数值矩阵第5列读 `Cycle_Index`，第78–82行对其去重、删除首末项，再以真实索引 j 定位记录。第87–92行只导出 k 对应的 VIQ/t 波形，第154行以 `(1:k)'` 重写 summary.cycle，未把 j 保存到 cycles 或 summary。各圈 Qd 与 summary 一致可以证明记录绑定，仍不能恢复被省略的仪器圈号、其间空缺和第一条真实循环。仅依赖四个 MATLAB 档案时，固定169颗均缺少这一证据，合同 D0-1 与 G 阻止正式 K100 冻结。现已发现并开始恢复原始 CSV，此缺口须按逐源真实恢复状态更新。[作者源码](https://raw.githubusercontent.com/chueh-ermon/BMS-autoanalysis/master/cell_analysis.m)

固定169队列继承了历史选择。旧源先删除5颗未观测到80%的 b1 电芯，再删除6颗 b3 噪声通道。源清单应保留全部185段、180颗原始物理电芯与11颗历史排除的来历，在固定169内继续避免按后缀长度二次筛选。新 `build_source_registry.py` 已保留 `outside_fixed169` 和历史选择字段。

本地 `BatteryLife/dataset/MATR` 为供应 PKL，`MATR_policy_metadata_20261029` 为先前策略恢复材料。完整新源位于研究目录的 `private/raw_sources/MATR`，目前四个 MATLAB 档案均已恢复，其完整字节数、全文件 SHA 和 HDF5 结构以各 receipt 为准。作者源码第92行将导出 I 定义为仪器 Current/1.1，第91行将相对仪器秒数除60保存为 t，第95–98行还执行负时间修正。回到 A 单位应保存原数值并显式乘1.1，t 的分钟含义及作者修正必须写入来源元数据，再复核 I–t–Q。直接把公开 I 数组标为 A 会留下量纲错误。原生 CSV 已使用 A/s，无需这些 MATLAB 导出倍率。[作者源码](https://raw.githubusercontent.com/chueh-ermon/BMS-autoanalysis/master/cell_analysis.m)

这四个档案为 `2017-05-12_batchdata_updated_struct_errorcorrect.mat`、`2017-06-30_batchdata_updated_struct_errorcorrect.mat`、`2018-04-12_batchdata_updated_struct_errorcorrect.mat`、`2019-01-24_batchdata_updated_struct_errorcorrect.mat`。MATLAB 下载与原始 CSV 映射恢复分别记录，主输入与目标直接取所有合格原生仪器记录，包括旧导出省略的完整首末圈。

## 真实来源的独立数值审计

`audit_source_records.py` 在最终阶段计量实现上实读9个源段共918条记录，覆盖各批至少一个源电芯、b1c0/b1c1/b1c18 与 b2c7 续测相关源段、b2c12/b2c44 的坏记录邻域。911条通过阶段容量和 Q(V) 数值规则，7条保留明确失败原因。计算时 measurements/raw_reader 的源码哈希保持一致。结果位于 `SOURCE_ORDER_PREFIX_AUDIT.json` 和 `source_record_measurement_audit.csv`，其中 cycle 坐标登记为作者 source-record 序列。早期910条版本保留于工程历史目录。

| 来源样本 | 审计记录 | 容量/Q(V) 有效 | 主要失败 |
|---|---|---|---|
| b1c0 | 100 | 98 | record0 短占位，record11 对应原生12圈的 Qd 零点0.526866 Ah未认证 |
| b1c1 | 100 | 99 | record0 短占位 |
| b1c18 | 100 | 98 | record0 短占位，record39 混合阶段/重置/量程 |
| b2c0 | 100 | 100 | — |
| b2c7 | 100 | 100 | — |
| b2c12 | 109 | 108 | record252 counter 零点与窗口支持 |
| b2c44 | 109 | 108 | record247 counter 零点与窗口支持 |
| b3c0 | 100 | 100 | — |
| b4c0 | 100 | 100 | record35 时间字段失败，设备 counter 容量与 Q(V) 有效 |

恢复后的负电流中位约 −4.400 A。合格记录 I–t 积分与 Qd 增量之比，b1/2 样本中位为1.00036–1.00070，b3/b4 分别为1.01132/1.01166。以明确保存的主阶段起止点为条件，独立 SI 积分数值复算对全部可比较记录一致，该检查只核积分算术，阶段识别由实际电学重放及回归控制核验。跨批闭合差按原数据保留。Q(V) 使用设备 counter 绝对锚点，另外的形态版在3.3 V置零。

## 原始仪器 CSV 的流式读取与绑定

`instrument_csv.py` 按原生 `Cycle_Index` 分组，跨 chunk 保留完整圈，原生 Current A 与 Test_Time s 原样保留，Data_Point、Step_Index、DateTime、temperature、IR 和 optional Aux_Voltage 均可追溯。首组、末组和缺圈保留，真实 Cycle_Index 不连续时保持间隙，倒序/非整数轴明确失败。Index0 仅标初始放电候选，未认证形成身份。原生记录没有 summary 容量，阶段计量直接读取设备 Qd 终点，summary 仅用于有该字段的导出源交叉核对。

11项人工流式性质与4批真实64 KiB CSV前缀 schema 检查通过。完整 b2c0 日志为328组、488030行、仪器轴0–327，MATLAB326条分别与原始1–326圈的 V/I/Qc/Qd/t 精确匹配，五字段全局最大绝对差均为0。初始0圈和完整末327圈保留，67/168/278圈的原生时间反转独立登记。身份由完整 CSV SHA、EDP 条码/channel 与 MATLAB MCOS 字符串解码核对。产物为 `b2c0_INSTRUMENT_BINDING_QA.json` 及逐圈 CSV。

完整 b1c0 日志为1191组、1904146行、仪器轴0–1190。MATLAB1189条中1188条与原始2–1189圈五字段精确匹配，source record0 的单点占位未匹配，真实原始1圈的1090点得以恢复。原始1190末圈136点继续按完整放电资格检查。Index0含607046点、约4.38小时、低倍率多阶段混合及一次原始时间重置，未认证形成循环，保存独立诊断与失败掩码。

完整174原生 CSV 已恢复，独立全行通道审计覆盖143396900行、2695个源文件字段，Data_Point非递增0次、Cycle_Index下降1次、原生DateTime逆序6次、Test_Time逆序500次。下降发生于b1c18，原生run0末691后重启run1的0..53。54组后段无4C放电，且电流/Step协议与源SDU不同，登记为非aging characterization，主aging终点691。全部原始native/run/行界仍保留。5组跨批续测真实末索引offset分别1190、1179、1177、1226、1227，初始化0组保留来源角色，后段正索引按真实attempt累计并保留缺圈。

174份原Metadata.csv共24字段及174份源SDU完整Step/Limit/Equation/Goto字符串已恢复并逐字段审计。设备ID前导0保留，后两批`03603603`和前两批`60369369`具备独立条码/通道绑定。Metadata起止时间与CSV DateTime的跨字段时钟对齐尚未认证，差值只作数值审计。Metadata的全0 resumed字段不足以反证原生重启。

最终原始记录140576条，合格regular容量140295条。其中6780条的native主D Step标签与SDU字面编号不同，涉及111颗及386条前缀记录，actual I/V/Q、完整4C放电和已登记量程规则全部通过。3条原生日志例外精确重放支持标签滞后的解释，全部6780条尚未逐条重读，控制系统根因未认证。`SOURCE_DISCHARGE_STEP_QA.json`保留`COMPLETE_WITH_SOURCE_STEP_LABEL_EXCEPTIONS`及`exact_step_identity=False`语义，采用实际电学阶段证据和源4C/2 V定义认证，完整保留文字标签异常。

## 固定物理模块与正交约定

API 为 `extract_prefix_features(q_absolute, curve_valid, actualcycleids, vgrid)`，返回每个有效原始前缀行及显式 `input_row_index`。数组分别为 `global_features[N,20]`、`local_residual[N,64]`、`local_features[N,8,26]`，每组均有同形布尔掩码。无效特征使用 NaN。模块只接受已由调用方认证阶段、单位、counter 零点和物理循环的 absolute Q(V)。

参考使用真实循环 1–10 中合格曲线的逐点中位数，至少5条，最多10条。此选择在计算真实源特征前固定，保证 K10 锚点参考不借用 K10 之后的记录。参考不足时给出 reference mask 和字段原因，电芯资格表继续保留。

参考导数固定 SG 窗口5、二阶多项式平滑后，按真实 V 网格作二阶边界梯度。稳健分解使用 Huber IRLS，阈值1.345、最多20次、MAD 尺度、1e-6 Ah 噪声下限，a 单位 Ah，b 单位 V。b 正号定义为其乘参考 dQ/dV 加到 ΔQ。导数 RMS 下限1e-8 Ah/V，缩放后设计矩阵 condition 上限1e6，近恒导数和病态权重均触发掩码。

**近似正交采用最终 IRLS 权重的内积**，保存64个权重、`weighted_normalized_projection`、`unweighted_projection`、重构误差、迭代次数和收敛标记。后者使用 `[1, derivative/derivative_RMS]` 的未加权平均投影，单位 Ah。固定20次迭代形成一个明确估计器，未收敛标记继续报告。主特征保留 IRLS 解。在同一两维基上再作未加权 OLS 投影会使最后系数代数上退回 OLS，因而独立数学自检仅将 OLS 作为对照。

8个 bin 每个保留有符号残差均值、RMS、残差平方的电压积分、绝对残差质心/峰位置、可用比例、高频噪声代理和完整 ΔQ 均值。峰/质心低于噪声下限时 masked。能量单位为 Ah²·V。噪声代理为 Q−SG(5,2) 的 bin 内 MAD，包含测量、插值和局部结构成分，其量纲与来源明确登记。

a/b 及局部 signed mean/RMS 使用最近10条合格观测的因果中位数、瞬态差、MAD、Theil–Sen 两两斜率中位数、末步斜率、斜率变化、方向一致性和支持数。所有斜率与变化率按真实循环差归一化。K10/K50/K100 锚点取 `[K−4,K]` 内最近合格圈，并保存实际圈号和支持，无后向未来借用。ICA/DVA、R 与极化认证默认 false，对应主 D0 输出列继续 missing。

## 实测数学 QA

运行 `verify_physics_features.py`，18项人工性质检查通过，结果位于 `results/unified_matr_seed42/PHYSICS_FEATURES_MATH_QA.json`。

| 检查 | 实测值或状态 |
|---|---|
| 已知 a/b 恢复最大绝对误差 | 1.69e-17 |
| ΔQ 重构最大误差 | 4.34e-19 Ah |
| 归一化加权投影最大值 | 9.71e-13 |
| 单点0.1 Ah 污染后的稳健系数误差 | 1.34e-7 |
| 同例 OLS 数学对照系数误差 | 9.53e-3 |
| 同例未加权平均投影 | 2.44e-3 Ah，按约定保存 |
| 增加40圈后缀并扰动为1e20 | 全部前缀特征精确一致 |
| 原始圈存在20、40、100的间隔 | 恢复真实斜率0.0001 Ah/cycle |
| 常导数、参考不足、坏点、来源失败、重复物理圈号 | 掩码或明确异常符合约定 |

这些结果认证数学实现和前缀隔离性质。真实数据的物理量程、阶段语义、计量零点和原始轴认证仍需 D0 集成后实读检查。

## 优势、风险与工程默认

优势是固定物理坐标、原始64维局部残差和字段掩码可联合追溯，稳健分解与其内积正交保持数学一致，真实间隔动力学可在缺圈时保留正确单位。模型比较共用已冻结输入与资格分母。

风险包括169队列的历史选择及测试暴露、带载响应与噪声代理的混合来源、近常导数下 a/b 的可辨识性、前10圈参考支持不足，以及只用seed42时训练随机性尚未测量。上述风险有显式字段和报告位置，任何性能结论仍需完整正式矩阵。

合同未指定精确 current/contiguity/integration 阈值、最低前缀支持数、SOH 主分母、参考圈选择、鲁棒算法数值、锚点时距、折数、验证比例和训练超参数。当前取值登记在 `configs/unified_matr_seed42.json`、本机器清单及 `physics_features.DEFAULT_CONFIG`，均属于源处理与评分之前的可逆工程选择。80条前缀支持只作支持率报告，合法参考与末端真实容量锚点按模型依赖保留掩码，分解病态仅影响对应特征。冻结后改变任何实质定义须新版本并保留旧记录。

数学自检结论置信度为**高**，依据为可重复的明确数值与断言。公开 summary.cycle 无法独立认证仪器原始圈轴的结论置信度为**高**，依据为作者明确的字段提取、选择和重编号代码。主队列174来源恢复、独立身份绑定、全行native转移审计、5组跨批续测和b1c18非aging重启角色已有证据。完整D0产物还须独立终验与实际训练loader读取核验，正式模型效果当前为**未评估**。

`verify_dataset.py` 对所有174真实CSV重新计算全文件SHA并以独立扫描器逐组核验native/run/全局轴和行界，核验完整元数据、目标/掩码、富集元数据、源标签例外、固定物理特征、因果隔离及真实训练折标准化。最后通过实际`train_seed42.load_frozen_data`的临时证书重放，再生成DATA-FREEZE。实际状态以`data/unified_matr_seed42/dataset_qa.json`、`dataset_qa_report.md`和`DATA_FREEZE.json`为准，本合同审查清单不会替代实际通过状态。

## 最终数据冻结实测

最终独立终验580项全部通过，并执行实际训练loader的只读临时证书重放后生成DATA-FREEZE。随后再次读取已存在证书，44个冻结文件的SHA全部重新核验通过。训练loader probe没有实例化模型、更新优化器或评分预测。

| 实测分母 | 数值 |
|---|---|
| 固定物理电芯 / 输入合格 | 169 / 169 |
| 原生source-record审计行 | 140576 |
| 合格regular容量记录 | 140295 |
| 原始1–100圈合格Q(V) | 16857 / 16900 |
| 全部有效未来SOH观测点 | 123431 |
| 200–500端点速率支持 | 151 |
| 500–1000端点速率支持 | 42 |
| 名义1.1 Ah分母EOL80事件 / 右删失 | 169 / 0 |
| 逐电芯最后有效真实圈范围 | 171–2238 |
| 原始设备03603603 / 60369369电芯数 | 85 / 84 |

DATA_FREEZE SHA256为`276016ea2f72c2d9c366c44b00906c80cf8234f741b48b60a6a2562c57fff4fb`。数据fingerprint SHA256为`04af53bc81d03e976b1f4e9dce776540f251bfc2d960311e0f0b3ffd80eedaec`。

EOL80按登记的名义1.1 Ah主分母计算，阈值为0.88 Ah。早期实际容量归一化的SOH同时保存，其事件统计须按对应定义另行计算，历史其它口径的事件数不可移用于本次主任务。数据终验结果尚未构成模型性能结论。

冻结后额外独立只读核对确认，169颗`cells.source_segments`逐一等于真实身份bindings，官方标签比较表的174个`(cell_uid,source_segment_id)`唯一且完整覆盖来源段。逐行官方life/coordinate与static审计一致，认证事件/首次crossing/最后有效圈与cells一致，官方标签均保持QA-only。该核对未修改冻结文件或模型。
