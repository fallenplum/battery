# SDU 新增快照完整元数据、信号与独立D0资格审计

审计日期为2026-11-15，数据来自本次新增的SDU.zip。该快照独立于已冻结的427个主要D0电池。

## 覆盖

压缩包有86个标准化电池PKL，共49,146圈，未压缩3,959,873,319字节。全部文件完整SHA-256、zip成员CRC、全部顶层元数据、每循环schema与全部数值点均完成检查。压缩包本次读取前后SHA-256一致性为True。

精确寿命标签命中70/86。其余文件仍保留信号，标签生成版本与当前快照的关联尚未独立证实。

## 元数据与来源

当前标准化文件中的材料、容量、电压限制、SOC和删圈列表属于供应元数据，逐文件完整值见sdu_cell_metadata.csv，所有取值及计数见sdu_family_summary.json。

|字段|完整取值及电池计数|
|---|---|
|cell_id|86种取值，完整逐电池列表在CSV/JSON|
|form_factor|[{"value": "cylindrical", "cells": 86}]|
|anode_material|[{"value": "graphite", "cells": 86}]|
|cathode_material|[{"value": "NMC_532", "cells": 86}]|
|electrolyte_material|[{"value": null, "cells": 86}]|
|nominal_capacity_in_Ah|[{"value": 2.4, "cells": 86}]|
|depth_of_charge|[{"value": 1.0, "cells": 86}]|
|depth_of_discharge|[{"value": 1.0, "cells": 86}]|
|already_spent_cycles|[{"value": 0, "cells": 86}]|
|max_voltage_limit_in_V|[{"value": 4.2, "cells": 86}]|
|min_voltage_limit_in_V|[{"value": 3, "cells": 86}]|
|max_current_limit_in_A|[{"value": null, "cells": 86}]|
|min_current_limit_in_A|[{"value": null, "cells": 86}]|
|reference|[{"value": null, "cells": 86}]|
|description|[{"value": null, "cells": 86}]|
|charge_protocol|[{"value": [], "cells": 86}]|
|discharge_protocol|[{"value": [], "cells": 86}]|
|SOC_interval|[{"value": [0, 1], "cells": 86}]|
|hardcoded_removed_indices|14种取值，完整逐电池列表在CSV/JSON|
|median_removed_indices|1种取值，完整逐电池列表在CSV/JSON|

官方源研究包含初次使用和第二寿命两个阶段。初次使用包含86颗NMC532/graphite电池、16种协议，诊断测试每4个工作循环执行，原始CSV含Battery_ID与Protocol_ID。标准化PKL中未发现Protocol_ID，IR字段的实际覆盖以本次扫描为准。官方文件名编号与论文Battery_ID不同，原始文件名61/62缺失的说明无法直接用于否定标准化Battery_61/62身份。[SDU官方数据记录](https://zenodo.org/records/14859405)

本地preprocess_SDU.py按Test_Time排序并重组Cycle_Index，按硬编码容量/循环窗口删圈，将指定诊断圈替换为最近普通圈的完整V/I/Q/t/T，再连续重编号并调用统一时间修复。hardcoded_removed_indices是删圈前列表位置，替换圈位置未单独保存。当前源码与该zip的生成版本关联仍未证实。源码行号和SHA见sdu_preprocessing_lineage.json。

## 全通道数值质量

|通道|有限/总点数|有限值范围|None/缺键圈|NaN点|
|---|---|---|---|---|
|current_in_A|73,264,808/73,264,808|-7.20098至4.800911|0|0|
|voltage_in_V|73,264,808/73,264,808|1.999305至4.300264|0|0|
|charge_capacity_in_Ah|73,264,808/73,264,808|0.0至2.640483852465697|0|0|
|discharge_capacity_in_Ah|73,264,808/73,264,808|0.0至2.6291618701689172|0|0|
|time_in_s|73,264,808/73,264,808|0.0至37372824.00290008|0|0|
|temperature_in_C|73,264,808/73,264,808|18.6832504280175至58.9174182385098|0|0|
|internal_resistance_in_ohm|0/0|None至None|49,146|0|

循环与时间问题的精确计数如下。

```json
{
  "negative_time_steps": 0,
  "repeated_time_steps": 0,
  "exact_repeated_V_I_Qc_Qd_cycles": 9199
}
```

重复VIQ指纹使用完整V/I/Qc/Qd数组，时间与温度未加入。后续同指纹记录从本次D0特征和目标候选中排除。首个保留记录的原始身份仍可能受源代码诊断替换影响，因此该排除不能认证真实物理循环。

## 寿命标签、事件与删失

寿命标量大于当前循环记录条数的电池为6个，大于最大供应循环编号的电池为6个。标签不用于生成已观察事件。

未来容量由放电电流阶段确认后的容量最大值减去有限全圈容量基线，再除以声明容量。排除重复VIQ后计算200–500、500–1000供应循环号的端点窗口平均斜率，EOL80及稳定膝点候选。EOL字段区分none、interval、right、left、unknown，首次观测已低于阈值单独按left报告。

## 独立D0资格与支持

|供应前缀K|编号1至K齐全/电池|至少2条合格观测曲线/电池|真实物理循环认证|真实ICA/DVA|
|---|---|---|---|---|
|10|86|70|0|0/0|
|50|86|70|0|0/0|
|100|73|70|0|0/0|

独立新cohort的200–500斜率候选有36颗，500–1000斜率候选有24颗，稳定膝点候选有67颗。EOL删失类别为{'none': 76, 'right': 9, 'interval': 1}。同一电池的严格物理认证交集为0，完整观测候选交集见sdu_feature_coverage.csv。

曲线设置为放电、共同电压窗口3.4–4.1V、前缀K10/50/100。每个K的参考电流仅由该K内未重复供应记录的前五圈估计。V/I/Q形状与IC形状代理按run_d0已有逻辑计算，所有输入均限制在对应前缀。未来目标留在独立结果列，严格物理目标和真实ICA/DVA均未认证。

## 优势与劣势（风险）

优势为同一NMC532/graphite体系、温度时序和较长轨迹，且新增86个电池的字段与有效支持可完整追踪。劣势（风险）为Protocol_ID、诊断身份和原始循环映射缺失，删圈与整圈替换改变供应坐标，时间修复影响真实持续时长。当前文件未包含原始两阶段CSV，未来政策状态记为unknown。

## 置信度评级

当前zip数量、字段、数值、哈希、标签精确匹配和供应坐标候选支持的置信度为高。源码行为与zip生成关联、真实物理循环和未来策略认证的置信度为低，需要原始CSV对照。

## 复现与边界

使用scripts/audit_sdu.py传入source、labels、preprocessor、output、private-output和report参数。扫描逻辑来自上一轮完整BatteryLife审计，记录了参考实现SHA及当前脚本/依赖SHA。提取的PKL和逐循环中间记录放在private-output，公开结果包含全部元数据、质量、资格和目标支持表。SDU未并入已冻结427个主要D0或主实验B。

## 拓展思考

原始诊断循环包含与日常工作循环不同的测量条件。恢复其身份和Protocol_ID后，可以独立审视诊断容量、工况与早期响应的对应关系。

## 你可能想知道

官方提供初次使用与第二寿命两个原始压缩包。当前新增SDU.zip是标准化电池文件集合，已核验的86个PKL本身不能建立两阶段原始轨迹的一一对应。

## 最终D0函数版本复核

完整信号审计保留最初执行时的依赖记录。完成后又以当前共享run_d0和physics_targets函数独立补算86颗电池的D0候选，并逐个重算PKL SHA256。依赖版本与检查见sdu_D0_refresh_provenance.json，最终资格/特征/目标CSV均来自这次补算，完整通道数值统计继续使用已完成的原始全扫。

本次新增的SDU元数据包含20个非cycle_data顶层字段，逐字段覆盖见sdu_metadata_field_coverage.csv。温度全部有限，内阻全部为空，充放电协议全部为空列表。

声明电压限制为3.0–4.2V，实测全局范围为1.999305–4.300264V。供应值与声明边界的偏离幅度已逐电池保存，采用零容差记录数值，具体实验有效范围需按原始16种协议解释。

完整复现先执行scripts/audit_sdu.py，再执行scripts/refresh_sdu_d0.py。两个命令的参数均使用显式输入和输出路径。

## 独立交叉核验

20项独立检查全部通过，记录在sdu_independent_verification.json。除全部公开表与逐电池证据的重算外，选择4个不同极值/长度来源文件重新解码六个通道，点数、finite数、min/max与原扫描一致，SHA256均重新匹配。代表电池删除全部K100之后的后缀后，K10/50/100早期特征完全一致。完整ZIP SHA256再次核对一致。

实际字段类型和单位解释见sdu_data_dictionary.json。14颗电池记录35个删圈前原始序号，83颗出现后续重复VIQ，总计9199条，这些后续重复项已从候选特征与目标中排除。
