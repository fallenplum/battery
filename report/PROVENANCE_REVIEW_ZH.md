# D0 来源、身份与原始循环核验

审计日期为2026-11-15。本报告复用既有P0身份链，并新增官方HUST原始记录恢复及全77电池核验。

## 核验结论

| 对象 | 当前覆盖 | 来源认证结果 |
|---|---|---|
| 旧576队列 | 10家族、576文件 | 全部当前PKL SHA与历史绑定相同，576个trace路径存在，9份P0来源manifest SHA全部匹配 |
| HUST | 77原始电池、146122个原始循环 | 官方完整归档SHA与全部77成员CRC通过，原始循环号和Cycle number列逐圈一致 |
| HUST早期映射 | 77电池各前100条标准化保留记录 | V逐点完全相同，I逐点等于原始mA/1000，原始第1–100圈均有完整6阶段 |
| MATR | 169标准化电池，旧队列103 | 原始policy精确覆盖124文件，384个HDF5分块只提供部分元数据，原始全波形及原始循环转换尚未重放 |
| Stanford_2 | 181标准化电池，旧队列101 | 官方老化数据自身为BEEP结构化插值，原始工厂形成记录另存，native老化采样与原始cycle-index映射尚未恢复 |

旧576的全部ID和当前SHA连接见`old576_current_data_identity.csv`。该映射确认现存文件与旧队列的身份关系，数据的历史研究曝光仍需保留。

## HUST恢复与安全解析

官方版本为Mendeley nsc7hnsg4s/2。唯一归档`our_data.zip`为1188136932字节，完整SHA256为`071d24617153693b0d29059568525e620f6af6512acc9d00c98c7adcf15125db`。先用HTTP206读取尾64KiB确认78项目录，再恢复整包，77个PKL总解压体积4245730747字节，全部CRC通过。[官方数据记录](https://data.mendeley.com/datasets/nsc7hnsg4s/2)

初始直接`pickle.loads`方案被automatic approval review拒绝，原因是外部pickle可以调用任意构造器。实际采用`pickletools`非执行符号解析，GLOBAL/REDUCE/NEWOBJ/BUILD均保留为数据节点，只显式读取数值buffer和primitive状态字符串。任意GLOBAL调用为0，针对任意全局调用、错误buffer长度和非primitive object payload的3项验证全部通过。

原始文件保存于实验private目录。可推送仓库只保存恢复代码、官方源注册、逐电池资格表及汇总证据。

## HUST原始字段与循环坐标

每个原始电池有`rul`、`dq`、`data`。本次来源认证使用`data`中的显式记录，未将rul或dq标签直接当作观测EOL。

每圈六列为`Status`、`Cycle number`、`Current (mA)`、`Voltage (V)`、`Capacity (mAh)`、`Time (s)`。全部146122圈均保留相同完整顺序，内容为CC充电、CC-CV充电和四段放电。原始cycle keys与每圈Cycle number列全部一致，圈内负时间步为0，相邻重复时间步共4个。

原始时间每圈从0开始，只支持cycle-relative秒。公开循环编号能作为原始实验的循环坐标，制造形成前史和真实日历时间仍缺少记录。

`HUST_7-5`原始有1875圈，标准化删去前2圈。其标准化前100条对应原始3–102圈，因此物理K100输入最多包含原始<=100的98条标准化保留记录。

原始Capacity列为净电荷计数，充电递增、放电递减，四段放电连续。放电容量应根据放电起点减去末点构造，直接取该列最大值会混淆充电端点与放电量。现存标准化Qc/Qd由电流积分重建，记录定义需要保留。

## 未来工况证据与冲突

官方记录说明77个LFP/graphite电池有相同充电协议、不同多段放电协议，环境设定为30°C。当前原始列没有电芯温度实测。[官方实验说明](https://data.mendeley.com/datasets/nsc7hnsg4s/2)

原始全部77电池的最后放电段跨循环电流中位数均约2.00C，当前BatteryLife预处理代码却把最终段写为1C。前三段与DISCHARGE_RATES字典一致，阶段中位数偏差大于0.05C的循环数为0。CC充电段仅1个循环的中位数偏离5C超过0.05C。

完整suffix的Status与电流统计用于来源和工况审计，未作为早期模型特征。官方固定实验设计、实测稳定工况和预测时是否持有原始控制脚本仍需分别记录。`policy_scheduled_evidence.csv`保留声明与原始观测冲突，当前候选按observed_only保守标注。RSC官方补充材料及作者稿返回403，尚未用原始控制器脚本闭合最终2C的事先登记证据。

## 既有P0与E盘归档

旧P0的23份raw probes实际读取BatteryLife供应方标准化PKL。576份trace及处理时间缓存提供强的retained-record回放证据，历史manifest明确`physical_cycle_axis_certified=false`和`processed_time_is_calendar_time=false`。

本次重新计算现有physical input和processed time cache的SHA，与登记一致。E盘纯输入归档330182867字节已完整SHA核验为`6bc4cac9d11ff91cc829f8a3f155db9f498730b7db702552cc596b8c21eb6767`。已登记source-internal MANIFEST的SHA也与P0索引一致。上述缓存身份能够复用，MATR/Stanford原始物理循环认证仍有各自缺口。

## Stanford与MATR恢复边界

Stanford官方说明老化部分为BEEP结构化/插值曲线，另含raw formation数据。直接重新下载当前官方包仍需保留这层插值属性。原始cycle_index和形成圈转换可通过原始deposition重放建立映射，native高频老化采样是否公开需要单独确认。[官方Stanford数据说明](https://data.matr.io/8/)

MATR三批policy已有可信来源索引和跨批续测关系，可复用其元数据认证。现存45个b4文件缺少这套三批policy覆盖，部分HDF5 byte-range缓存无法替代原始波形。恢复原始对应批次后，应按original/physical/processed三类ID重放删首圈、跳坏圈与续测拼接。

## 优势与劣势（风险）

优势是旧576身份链可完整关联，HUST已获得官方原始全包和逐圈显式阶段，从源头补齐真实循环编号与早期VI信号对应。

劣势（风险）是HUST当前协议元数据存在1C/2C冲突，日历时间、工厂形成前史和实测温度仍缺失。MATR和Stanford的原始物理循环映射仍有技术缺口，ICA/DVA稳定性和未来目标资格需由主D0继续裁决。

## 置信度评级

文件SHA、CRC、schema、Cycle number、Status顺序和前100逐点VI一致性的置信度为高。原始公开循环编号与制造全过程累计老化的对应关系为中，缺少形成前史。以现有声明判定每颗电池预测时拥有完整未来控制脚本的置信度为低，实验设计证据与实际观测证据已经分开保留。

## 拓展思考

HUST原始Status和净容量计数可用于核对分阶段Q(V)的窗口和方向，并与标准化电流积分容量进行独立定义对照。

## 你可能想知道

原始数据中的rul和dq已有科学计算口径，但观测阈值事件、删失和膝点仍需要按本次预注册规则从有效容量轨迹构造。


## 原始容量目标增量核验

官方HUST的146122条原始容量记录均通过计数方向/阶段连续性检查，77颗电池的200–500及500–1000循环平均速率均可用。原始counter SOH最小0.8000045025，精确0.8阈值下事件为0、右删失77，未把rul/dq当事件。provided/raw每电池中位比为1.00970–1.02402，阶段边界采样和积分定义差异已记录。详情见仓库`report/HUST_RAW_CAPACITY_TARGET_REVIEW_ZH.md`和`results/d0/HUST_RAW_CAPACITY_VERIFICATION.json`。
