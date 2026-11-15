# MATR 原生 run、协议与时间字段审查

## 原始轴与测量依赖

原生 CSV 的 `Cycle_Index` 是仪器程序的循环索引。重复执行程序中的小步骤也可能递增该索引，因此需同时保留 run 身份与实际协议。主任务采用已知老化方案中的原生 regular aging 循环，原始本地索引、数据行位置、重启、初始化和后续表征记录全部留存。

设备 Qd 是直接计量的电荷 counter。认证完整、符号一致的放电阶段及 counter 零点之后，容量和 Q(V) 依赖 I/V/Qd 与采集顺序。圈内时间用于积分闭合、瞬态及时间导数，时间失效应留下 `time_valid=False`、`integration_valid=False` 和字段警告。经独立认证的 counter/Q(V) 可继续保持其测量资格。原数据和失败时间值原样保留。

原生 CSV 的 Current 为 A、Test_Time 为秒。公开 MATLAB 导出将 I 除1.1、相对秒数除60，并在部分负时间记录执行修正。原生数据直接采用 A/s，MATLAB 来源审计单独登记其倍率。[作者导出代码](https://raw.githubusercontent.com/chueh-ermon/BMS-autoanalysis/master/cell_analysis.m)

## b1c18 的同文件重启

独立的 CSV 全行五列扫描与主流式解析器均复现以下转移。

| 字段 | 前一行 | 后一行 |
|---|---|---|
| 数据行/Data_Point | 964481 | 964482 |
| Cycle_Index | 691 | 0 |
| Step_Index | 11 | 0 |
| DateTime | 1496760566.4781 | 1496808858.1636 |
| Test_Time | 197614.2787 | 245905.9642 |

run0 包含原生0–691共692组、964482行，run1包含原生0–53共54组、151699行。时间和采集顺序支持这是程序重启。独立 run 使同号原生圈各自保留，避免作者 `unique(Cycle_Index)` 再 `find(Cycle_Index==j)` 将不同执行时期的同号段拼入一条记录。公开 MATLAB 的 summary 又被改写成1:k，须从原生日志认证主时间轴。[作者导出代码](https://raw.githubusercontent.com/chueh-ermon/BMS-autoanalysis/master/cell_analysis.m)

## 重启后的协议身份

b1c18 原始老化 schedule 共13个步骤，CC1 设定5.94 A、CC2设定3.3 A，Step10 放电设定−4.4 A。官方 test metadata 的 `last_end_datetime` 原值为1496760533，其与原生 CSV DateTime 的时钟、时区及导出对齐尚未认证，48325.16秒的跨字段差值保留为数值审计。同一 CSV 的 DateTime 在run0末1496760566.4781至run1始1496808858.1636正向间隔48291.69秒，可独立支持执行顺序。[官方实验方案](https://data.matr.io/1/api/v1/file/5c86c288fa2ede00015de122/download)、[官方测试元数据](https://data.matr.io/1/api/v1/file/5c86c288fa2ede00015de125/download)

run1 的完整54组数据另有以下实测结构。

- native0 为约±1.1 A的初始化/充放电混合阶段，使用Step0–6。
- native1–39 为约+0.275 A充电及静置，逐组ΔQc约0.0275 Ah，Qd固定在0.87117839 Ah。
- native40 含约±1.1 A的混合表征过程。
- native41–49 为约+0.55 A充电及静置，逐组ΔQc仍约0.0275 Ah。
- native50–53 近零电流，末组电压异常至6.606 V，原值及资格原因继续保留。

全部54组均无主任务的4C放电。结合原schedule和实际Step/VIQ，其角色登记为**重启后非老化表征**有直接来源支持。具体诊断方法名称尚缺独立认证。主 aging `original_cycle_id` 对该run记为missing，并保留原始run/native/row、generic累计索引和协议资格原因。主原生 aging run 保持1–691，后续充电增量保存在独立的native run轴。

逐组统计在 `results/unified_matr_seed42/b1c18_POST_RESTART_PROTOCOL_AUDIT.json`。官方两个文件已完整下载并记录 SHA256，元数据538字节，schedule19480字节。

## 跨批续测边界

五组已知续测的 barcode、channel、官方别名和原生时间顺序均独立核对。前段原生实际末索引依次为1190、1179、1177、1226、1227。后段从0重新初始化，native1开始为主方案合格放电。累计regular轴显式保留前段实际末attempt和缺圈槽位。

b1c3 的末1226有完整放电，Qd为0.96796095 Ah，应保留该真实观测。其他四颗末记录为未完成放电的attempt，保留索引与测量mask。b1c4 在native1029有一次约100微秒的DateTime逆序，同时Test_Time重置，Data_Point仍严格递增。该圈的时间资格单独记录，跨批边界本身仍有约377283秒正向间隔。

## 优势与风险

优势是输入窗口、未来目标与原生执行顺序可追溯，时钟故障、协议变化与counter质量分别有对应字段。早期run0数据不会因后续表征支持长度改变，实际完整末圈也能恢复。

风险是原生程序计数跨重启后需要明确的重建约定，协议变更必须逐源有证据，公开导出记录可能包含不同时期同号片段。最终独立数据终验580项全部通过，实际训练loader读取核验通过，44个冻结文件已完成第二次SHA核验。主矩阵的seed42模型效果仍由后续正式实验评估。对上述已实读的字段与程序角色判断，置信度为**高**。对具体电化学诊断名称，置信度为**低**。

最终169颗均有合法参考及真实末端输入锚点，前100真实圈共16857条合格Q(V)，全部未来有效SOH观测123431点。完整原始记录、无效圈、缺圈和后续非aging表征角色继续保留。6780条合格主D记录存在原生Step标签与SDU字面编号差异，111颗及386条前缀涉及该异常，实际I/V/Q、完整4C放电和counter零点支持保留。三条原生日志反例精确重放，全部例外尚未逐条重读，控制系统根因尚未认证。完整源审计状态明确记录为`COMPLETE_WITH_SOURCE_STEP_LABEL_EXCEPTIONS`。
