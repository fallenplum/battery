# DRS-PEFT 数据资格、输入语义与 Episode 冻结

核验日期为2026-11-17。本报告覆盖P0数据资格、实际可用数组、严格身份划分和归一化。P0训练与科学成绩由模型运行记录及P0_REPORT_ZH.md给出。

## 三域可用输入与全轨迹标签

| 数据域 | 独立电池 | 目标坐标 | 额定容量Ah | 合法放电窗V | 合格前缀曲线 | 未来SOH观测点 | 当前全域最大观测坐标 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MATR | 169 | original_instrument_cycle | 1.1 | 2.8–3.3 | 16857/16900 | 123431 | 2238 |
| HUST | 77 | original_instrument_cycle | 1.1 | 2.8–3.1 | 7698/7700 | 138422 | 2689 |
| Stanford_2 | 181 | supplied_cycle_number | 0.24 | 3.4–4.1 | 18092/18100 | 170397 | 1624 |

MATR和HUST通过既有n012资格加载器重新验证冻结数据文件与原始循环资格。HUST使用真实约2C末段放电窗口，保留源设计末段倍率与实测前缀倍率的差异。两域原文件、DATA_FREEZE与历史结果保持原始字节。

Stanford全部181个当前PKL逐文件重算SHA256，与D0冻结记录全部一致，共20364177417字节。随后逐文件读取，限定前缀构建完成后独立提取未来标签，一次只保留一颗原始电池对象。181个供应UID与181个作者硬件标识哈希一一对应，源表连接保留183行中未在本地出现的197与298。当前形成六参数精确元组有63组，55组三颗、8组两颗。

Stanford全部181颗具有供应编号1–100，173颗有100条有效电学曲线、8颗有99条。完整顶层非cycle_data元数据18字段逐电池保存在新缓存cell_metadata.csv的原始metadata JSON，包括pouch、graphite、LiNi0.5Mn0.3Co0.2O2、0.24Ah、3–4.4V声明与已知1C充电/0.75C放电设计。形成温度与30°C老化温箱设定分别记录。电芯实测温度、真实原始仪器循环映射、具体BEEP生成版本与电极机制真值保留未认证状态。

当前Stanford供应记录采用已插值并重编号坐标，全部指标必须保留supplied_cycle_number。供应标签定义为相应电流相位内的最大放电Ah减去该条记录有限容量基线，再除声明0.24Ah，合法范围为0–1.5之间。源原论文使用的初始老化容量80%寿命参考另有定义。本任务名义SOH80只按当前统一名义容量口径记录，HUST全部77颗右删失，不开展无事件EOL精度评价。

## 合法前缀接口

每域缓存包含raw_viq[N,100,64,3]、point_mask、cycle_ids、真实drift_q及其mask、28维具名policy/mask、SOH100观察锚点、global_features[N,100,20]、local_residual[N,100,64]、local_features[N,100,8,26]及各自mask。

所有原始电学数组保留V、A、Ah。真实drift定义为同一电池Q_n(V)减去1–10圈有效曲线的逐电压点中位参考，至少需要5条支持。原N012第四通道源于Q与参考尺度归一化Q的差值，DRS采用单独构造的跨循环响应差。三域全部有效原始点均具有足够drift参考，point_mask=True且drift_mask=False的点数为0。

Stanford重用既有current_reference、observed_curve和capacity资格逻辑，并从同一合格恒流放电段同步插值实际V、I、Q。64点窗无外推，curve资格与VIQ内容量重建逐值核对。物理分解使用相同数学提取器与Stanford独立3.4–4.1V配置，属于供应曲线上的响应分解。三个域的同索引bin保留各自真实电位，形状统计的相对窗坐标转换具有明确记录。

模型默认读取SourceOnlyScaler.transform的canonical x[N,100,64,4]，通道依次为真实V、带符号I/nominal_Ah倍率、Q/nominal_Ah、真实drift/nominal_Ah。canonical x保留电压物理尺度。可选x_z另用源训练统计与固定物理标准差下限处理，默认Bridge绝对响应支路使用canonical x。未来标签、未来观测mask、单cell末次循环、EOL、供应完整轨迹长度和UID均未进入transform的模型输入字段。

预测查询统一为101–5000整数网格，5001–6000可作为无标签数值诊断。当前三域全体合格未来观测都被主网格覆盖。未观察位置保留mask，禁止将占位0当成SOH真值。单cell实际末次观测只参与评分mask与审计。

## 28维政策语义与源训练归一化

共同policy按名称映射，保留名义容量、窗上下界、温箱设计、恒流充电阶段数、四个充电倍率、末段放电设计倍率、已注册电压限值、SOC切换与终点、topoff、六项形成先验、HUST四段放电设计倍率、两项前缀实测倍率及设计/观测冲突标记。字段与单位由POLICY_COLUMNS和POLICY_UNITS直接导出，标准差物理下限见POLICY_PHYSICAL_SCALES。

未知字段采用0值与False mask。MATR四段策略45颗的first_switch_soc缺失标志已转为False mask，124颗有该字段真实支持。缺失快充阶段的原显式标志同样转换为mask。HUST原设计与前缀观测分别保留，未猜测原数据没有的电压或SOC参数。

Scaler只在源训练85颗电池拟合。源全缺失列在所有域统一输出0且maskFalse，避免将目标独有形成字段当成已学习方向。source policy支持13维，HUST在该已支持子集内有8维、Stanford有7维。名义容量标准差下限1Ah、电压1V、温度10°C、倍率1C、静置72h等物理尺度使源常量字段具有有限跨域尺度，Stanford0.24Ah不会被1e-5经验std下限放大到极端z值。每cell的I/Q单位转换仍使用其已知名义容量。

原始global/local数组保留原单位。其可选*_z先将Ah类量除名义容量、Ah²类量除容量平方、b_V类量除注册窗宽，质心与峰电位转为明确的相对窗坐标，再拟合源训练统计。所有统计量、支持数、物理尺度和训练UID通过to_dict保存。可用的元数据与真正进入模型的字段分别可核查。

## 严格组互斥划分与支持分母

MATR169颗有77个exact protocol groups，组大小分布为1颗43组、2颗17组、3颗3组、5颗9组、6颗1组、7颗1组、8颗2组、9颗1组。冻结source train/meta-validation/target为47/15/15组、85/30/54颗电池。所有角色完整组互斥。目标15组纳入4个至少7颗的协议，仅以事前身份和组计数确定，其余组按seed42 SHA排序。旧unseen_protocol_sensitivity的train与validation有23个协议交集，本轮使用新划分。

| 协议A的shot K | 全169队列Q2可行组上限 | 新target15组内Q2可行组 | 固定Query电池数 | 注册支持draw | 实际不同支持集合 |
| --- | --- | --- | --- | --- | --- |
| 0 | 34 | 9 | 18 | 45 | 9 |
| 1 | 17 | 6 | 12 | 30 | 21 |
| 3 | 14 | 6 | 12 | 30 | 20 |
| 5 | 4 | 4 | 8 | 20 | 13 |
| 10 | 0 | 0 | 0 | 0 | 0 |

每个目标协议Query固定2颗，6个singleton协议明确标为NOT_FEASIBLE_Q2。不存在默认降成Q1的替换。exact协议最大组只有9颗，协议A的10-shot明确NOT_FEASIBLE。跨K主曲线可使用四个K5可行协议的共同8颗Query，全部可行episode结果另报真实分母。

跨域Stanford固定91颗Query、90颗Support候选，HUST固定39颗Query、38颗候选。Query对所有K、方法与支持draw完全相同。K=0只有一个真实空支持集合，正shot的5次注册draw在两个跨域任务中均得到5个不同支持集合。支持排序在每个draw内对K嵌套。重复支持集合具有同一指纹与duplicate_of_draw，适配checkpoint可以复用，重复条目不作为独立支持实验。

模型训练种子与episode种子分别记录。episode种子42，pilot种子42，后续正式多种子按新合同预注册为42/123/2026。主episode合同SHA256为1bc2f831eebc70998c73c6224773e590754c23e8abc395b0f3dbe597b4ff80c6。拆分仅读取UID和已知设计组字段，未使用未来SOH或目标误差。

## 运行与验证

```powershell
python -m scripts.drs_peft.episodes
python -m scripts.drs_peft.data --domain MATR
python -m scripts.drs_peft.data --domain HUST
python -m scripts.drs_peft.data --domain Stanford_2
python -m pytest tests/drs_peft/test_data.py -q
```

缓存位于runtime/drs_peft/data_v1，各域有arrays.npz、cell_metadata.csv、adapter_source.py和COMPLETE.json。缓存加载校验全部文件SHA256，NPZ禁用pickle对象。其体积适合向计算服务器复制，20.36GB原始Stanford数据继续保存在本地。显式--rebuild仅用于新版本数据构建与工程修复，正式阶段使用已锁缓存。

八项真实接口/工程回归测试通过，覆盖严格protocol互斥、重复支持分母、真实参考差、canonical物理尺度、源训练标准化、未见政策drop、名义容量常量尺度，以及Stanford后缀极值/重排/删除对前缀的独立性。测试中的合成曲线只用于功能验证，性能表需要真实训练产物。

附加读取真实Stanford_Nova_Regular_100，核验源SHA为1be1a5f5b69dc92fc89928272203e9c314149b0d4b53987d0bad0bf671e858f0。将全部后缀替换为带极端值、非有限值、乱序及超出主网格的供应坐标后，所有早期V/I/Q、drift、policy、anchor、global/local值与mask逐值相同。重新抽取的raw_viq与真实drift也与冻结新缓存逐值相同。

工程准备期间修正过一次Unicode UID存储类型，禁止object NPZ；随后修正MATR first_switch_soc缺失mask。最终三域缓存均由同一data.py SHA256 4dc110c68ed99a28ef61dfeb0bad50951d4855a3fb17b29dabc86d4b9ed5f585生成，并保存该源码字节快照。

## 优势、风险与置信度

优势为全物理UID隔离、支持重复分母明确、源训练统计可复核、固定全域查询网格与真实跨循环差。数据、身份和当前数值数组的执行核验置信度为高。

风险包括所有队列的历史研究暴露、Stanford插值与供应坐标版本、HUST完整容量测量算法的独立校准缺口、exact协议高shot支持有限、目标专有政策维度在源阶段未学习，以及跨域化学体系/电压窗/倍率差异。真实电极级机制解释的证据强度为低。SDU仍保持候选身份，原始循环/协议身份缺口需要独立修复。

## 拓展思考

few-shot增益需对应独立支持电池和相同Query，重复循环观测只能增加同一颗支持电池的标签密度。可将元验证30颗构造独立的pooled伪目标协议以选择K10适配预算。

## 你可能想知道

Stanford当前可执行明确声明的供应坐标预测。它的原始仪器周期认证与未来观测SOH任务资格分别保留，具体物理周期映射继续标记未认证。
