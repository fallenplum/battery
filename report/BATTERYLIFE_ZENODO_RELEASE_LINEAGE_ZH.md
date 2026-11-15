# BatteryLife处理版数据、Zenodo版本与预处理源码来源补充

## 已确认的来源

**用户提供的下载出处与BatteryLife官方文档一致，官方预处理源码和论文预处理说明均已定位。** 本地`D:\develop\patchtst\BatteryLife\README.md`第32–35行介绍处理版下载，第35行直接指向Zenodo记录18646655。第93–101行给出预处理入口，并说明下载处理版后可以跳过本地预处理。第116行说明把下载的数据集目录与Life labels放入`dataset`。这些信息可认证该路径属于官方处理版使用流程。[官方仓库README](https://github.com/Ruifeng-Tan/BatteryLife)

本报告于2026-11-15补充来源证据，核对采用本地文件、只读Git对象和官方GitHub、Zenodo、arXiv页面。公开仓库及Raw链接的检索缓存呈现过不同版本，以下版本判断采用明确的record ID和本地Git commit。

## 处理版发布记录

| 处理版 | 官方记录 | 发布时间 | 已直接核验的身份 |
|---|---|---|---|
| v10 | [Zenodo 18646655](https://zenodo.org/records/18646655) | 2026-02-24 | 本地README链接的BatteryLife_Processed版本 |
| v11 | [Zenodo 19688272](https://zenodo.org/records/19688272) | 2026-04-22 | 同一处理版系列的后续记录 |
| v12 | [Zenodo 21149533](https://zenodo.org/records/21149533) | 2026-07-03 | 本轮联网GitHub HTML页面链接的后续记录 |

三个固定record的官方文件清单均包含以下档案，并公布相同MD5。

| 文件 | v10、v11、v12共同公布的MD5 |
|---|---|
| HUST.zip | `de6b6d1b0b20616fbb96c72f3231c082` |
| MATR.zip | `83a1528858b9e1b7b6886757bb561669` |
| Stanford.zip | `6d6892eb5bd1b836635e5786c2b90c6a` |
| Stanford_2.zip | `3484565dc7c1dd2baa3df02352bbe8a5` |
| SDU.zip | `6d557d6e74e7b7cf13f54a9cedd7b38b` |

这些版本中的若干其他文件发生过变化，以上五个档案的官方校验值保持一致。处理版版本号与单个子数据集档案的内容版本需要同时登记。[v10文件清单](https://zenodo.org/records/18646655)、[v11文件清单](https://zenodo.org/records/19688272)、[v12文件清单](https://zenodo.org/records/21149533)

本轮主执行进程实测本地`D:\develop\patchtst\BatteryLife\dataset\SDU.zip`大小为2013763712字节，MD5为`6d557d6e74e7b7cf13f54a9cedd7b38b`，与三个record公布的SDU档案校验值一致。因此可以认证本地SDU档案与这一官方发布内容相符。三个record复用相同内容，用户当时下载的具体record仍需下载记录才能唯一定位。本地数据根目录当前保留的MATR、HUST、Stanford_2解包目录与本次SDU档案具有不同的留存形式，逐份PKL与原始压缩包member的完整绑定仍属独立核对范围。

## 论文关联与数据选择说明

本地README第3行及第185–196行关联KDD 2025论文，正式DOI为`10.1145/3711896.3737372`。本轮可访问的作者正文为arXiv `2502.18807v7`，修订日期2025-11-12。[论文记录](https://arxiv.org/abs/2502.18807v7)

**论文附录A.1明确介绍预处理。** 其内容包括按容量退化与寿命筛选电芯、对BatteryML标准化的七个数据集使用容量轨迹中值滤波检测并删除异常圈、对其他数据剔除RPT与形成圈，以及手工删除部分容量突变圈，最后保存为BatteryML统一格式。附录还明确论文使用Stanford首次发布版本，其表格为41颗。后续处理版加入Stanford_2和SDU的事实由官方发布更新记录补充，论文首次实验统计与后续下载包具有不同的版本范围。[附录A.1](https://arxiv.org/html/2502.18807v7#A1)、[官方版本更新记录](https://zenodo.org/records/18646655)

## 本地源码版本已经核定

本地Git HEAD为`db6b38a16aee19872c1dbb5dd71c745302e8ea41`，提交时间为2026-03-05T18:51:42+08:00，提交说明为`Update dataset statuses in README.md`，origin指向官方仓库。`git status`返回work-tree配置错误，本轮保持仓库配置原状，使用只读`git show HEAD:<path>`取得对象内容。

下列8个实际文件与HEAD对象逐一比较，**CRLF统一为LF后8/8完全一致**，文本diff均为空。实际字节SHA不同于Git对象字节SHA的原因是换行符。该核验为本地checkout的源码身份提供证据。公开Zenodo页面关联代码仓库，逐文件PKL生成时采用的commit、BatteryML安装版本与实际运行日志仍需发布构建记录绑定。

| 本地相对路径 | 当前实际文件SHA256 |
|---|---|
| README.md | `0cef93f190510f723ffa68ae53d0c3dbf14b7221806ef47af9dd3f49a7344f1f` |
| assets/Preprocess.md | `a592ed5e098681663c8d1aa99e3e32226f4c9a2090e23820e9ff539a654725df` |
| preprocess_scripts.py | `2e6c0c90aca6ec71f35f3a22cf0669b4e4867e93d89545bde72c7dfa97acb61f` |
| process_scripts/preprocess_MATR.py | `1834c209312f8b63e6f09c87ad2671e23befe3ec0d90afcafb4b417e462970ea` |
| process_scripts/preprocess_HUST.py | `88c2dc696d20b1fe08d56aa0d3f13783567775e72b14bbbfc08f5fa71040bd5e` |
| process_scripts/preprocess_Stanford.py | `e8969b29ffded9f9443b05e4e30e6add65af57c06657a9c95bd215e743e7c056` |
| process_scripts/preprocess_SDU.py | `a1c5c1f0e155b6e0c951d27be1376be8a718b74f59408c990af6a21cd0be8709` |
| process_scripts/time_normalization_utils.py | `d9a1b7f816ab532433ef99158cacc58f072d32c35bb774b6a7644013d1a57e93` |

四个数据集脚本及时间工具的最后内容提交也可从本地Git历史定位。

| 文件 | 最后修改提交 | 提交日期 |
|---|---|---|
| preprocess_MATR.py | `26bff53c9b8ea4b4f9e106c88e593d918959edbd` | 2025-12-01 |
| preprocess_HUST.py、preprocess_Stanford.py | `c62645a1e471d6c877183d92232d303df2fd75e3` | 2025-09-23 |
| preprocess_SDU.py | `bc689395b10b3fdefe801c18139dc6f02257de13` | 2025-10-29 |
| time_normalization_utils.py | `09ef198da24fbfbb92fb65d05de7d4e573f4bb7c` | 2026-01-09 |

## 预处理入口和依赖位置

本地根目录为`D:\develop\patchtst\BatteryLife`，以下行号按当前实际文件读取。

| 位置 | 已核实内容 |
|---|---|
| [README.md第93行](D:/develop/patchtst/BatteryLife/README.md:93) | 预处理章节与`python preprocess_scripts.py`入口 |
| [assets/Preprocess.md第15行](D:/develop/patchtst/BatteryLife/assets/Preprocess.md:15) | 将`process_scripts`全部文件复制到BatteryML的`batteryml/preprocess/`后安装 |
| [assets/Preprocess.md第28行](D:/develop/patchtst/BatteryLife/assets/Preprocess.md:28) | 原始数据准备目录`datasets/raw/` |
| [assets/Preprocess.md第57行](D:/develop/patchtst/BatteryLife/assets/Preprocess.md:57) | 执行入口，输出到`datasets/processed/` |
| [preprocess_scripts.py第22行](D:/develop/patchtst/BatteryLife/preprocess_scripts.py:22) | HUST CLI命令 |
| [preprocess_scripts.py第23行](D:/develop/patchtst/BatteryLife/preprocess_scripts.py:23) | MATR CLI命令 |
| [preprocess_scripts.py第28行](D:/develop/patchtst/BatteryLife/preprocess_scripts.py:28) | Stanford CLI命令 |

本地入口第19–36行共列出16条命令，SDU对应类文件已存在，当前批量命令列表尚未包含SDU调用。教程对BatteryML复制安装的要求意味着一次确切生成流程还包含安装目录中的实现与依赖状态。[官方预处理教程](https://github.com/Ruifeng-Tan/BatteryLife/blob/main/assets/Preprocess.md)

## 四个数据集脚本的可证处理行为

### MATR

[preprocess_MATR.py第20行](D:/develop/patchtst/BatteryLife/process_scripts/preprocess_MATR.py:20)列出四个MAT输入。[第116行](D:/develop/patchtst/BatteryLife/process_scripts/preprocess_MATR.py:116)开始批次清理，删除5颗未达既定退化条件的电芯，将5个batch2续测源段合到对应batch1电芯，并使用固定续测长度更新寿命标签。[第168行](D:/develop/patchtst/BatteryLife/process_scripts/preprocess_MATR.py:168)开始逐圈组织，跳过首记录，对b1c0、b1c18、b2c12、b2c44执行指定位置后的记录平移。[第194行](D:/develop/patchtst/BatteryLife/process_scripts/preprocess_MATR.py:194)把MAT时间乘60，赋予输出循环号，直接保存MAT的I/Qc/Qd；[第239行](D:/develop/patchtst/BatteryLife/process_scripts/preprocess_MATR.py:239)调用统一时间规范化，第255行开始列出6颗batch3排除电芯。公开脚本可解释这些选择、续测与记录重排行为。[官方MATR脚本](https://github.com/Ruifeng-Tan/BatteryLife/blob/main/process_scripts/preprocess_MATR.py)

循环号来自该导出与重排路径，其物理原始圈对应关系仍需源MAT及仪器记录映射。直接赋值的I字段也应结合MAT作者导出单位核对。本次已冻结研究采用独立认证的原生CSV轴与A/s字段，相关来源证据见已有仪器轴和单位审计。

### HUST

[preprocess_HUST.py第52行](D:/develop/patchtst/BatteryLife/process_scripts/preprocess_HUST.py:52)读取HUST原始电芯文件，逐原始记录取I/V/t。电流从mA除1000转换为A，Qc/Qd由第219行开始的`calc_Q`按电流符号和时间差计算。第64行赋循环号，HUST_7-5在[第74行](D:/develop/patchtst/BatteryLife/process_scripts/preprocess_HUST.py:74)删除前两条记录，第78行调用时间规范化。固定充放电设计元数据从第88行起写入。这些行为提供来源、单位、容量构造和特例处理的可读实现。[官方HUST脚本](https://github.com/Ruifeng-Tan/BatteryLife/blob/main/process_scripts/preprocess_HUST.py)

当前脚本先按源t积分容量，再规范化输出时间。因此物理I–t–Q闭合与时间动力学应结合该处理顺序解释。

### Stanford及Stanford_2

[preprocess_Stanford.py第25行](D:/develop/patchtst/BatteryLife/process_scripts/preprocess_Stanford.py:25)以`data.tar.gz`为输入，读取maccor JSON。[第72行](D:/develop/patchtst/BatteryLife/process_scripts/preprocess_Stanford.py:72)使用已有`cycles_interpolated`数据，按`test_time`和`cycle_index`组织I/V/Q/温度。第127–142行重排循环号、删除重排后编号为1的记录，其注释将该操作描述为删除首个形成圈，还删除部分微小负电流行，并按电流符号将相应容量分支归零。第111行调用统一时间规范化。[官方Stanford脚本](https://github.com/Ruifeng-Tan/BatteryLife/blob/main/process_scripts/preprocess_Stanford.py)

本地`dataset/Stanford_2`内保留`Stanford_Nova_Regular_100.pkl`等文件名，`Stanford_`前缀与该脚本的命名规则一致。v6更新明确加入完整Stanford_2，具体下载文件与哪一批原始JSON、哪次处理运行的生成绑定仍需逐文件证据。[官方发布更新记录](https://zenodo.org/records/18646655)

### SDU

[preprocess_SDU.py第22行](D:/develop/patchtst/BatteryLife/process_scripts/preprocess_SDU.py:22)读取`Primary_use_phase/*.csv`。[第76行](D:/develop/patchtst/BatteryLife/process_scripts/preprocess_SDU.py:76)按Test_Time排序并组织循环索引，第91行按组枚举记录，第101–110行以I–t积分构造容量。第127–192行执行指定电芯与区间的容量规则删除，再连续编号。

[第195行](D:/develop/patchtst/BatteryLife/process_scripts/preprocess_SDU.py:195)以平均负电流约−0.48 A识别一类诊断记录，73、74、75号电芯例外。第219–238行搜索最近的普通记录，可向左或向右搜索，并用邻居的整条V/I/t/Q/温度替换诊断记录。第241–243行再次连续编号，第265行规范化时间，第276–287行保存材料、容量及部分删除索引。该公开代码使诊断替换和重编号的实现可检查。上述规则在用户实际下载PKL中的具体应用仍由生成映射或逐值对照认证。[官方SDU脚本](https://github.com/Ruifeng-Tan/BatteryLife/blob/main/process_scripts/preprocess_SDU.py)

## 时间规范化与物理特征解释

[time_normalization_utils.py第29行](D:/develop/patchtst/BatteryLife/process_scripts/time_normalization_utils.py:29)为四个脚本共同调用的入口。[第122行](D:/develop/patchtst/BatteryLife/process_scripts/time_normalization_utils.py:122)开始修复圈内时间重置与大间隔，第151–207行将超过300秒的圈内间隔压缩到1秒，并处理识别到的时间重置。该工具在本地HEAD中已存在，8文件核对已证明其当前内容与该HEAD一致。

官方v10更新说明第8行也明确记录时间规范化及长间隔压缩。因而处理版的时间字段具有经过规范化的语义，物理瞬态、时间导数、真实持续时间和I–t积分特征需要沿该语义追溯。Q(V)形态、处理版循环序列变化及真实仪器时间动力学各自具有明确的测量范围。[官方v10更新说明](https://github.com/Ruifeng-Tan/BatteryLife/blob/main/assets/Version10_Update_Details.md)

## 认证范围与当前状态

| 问题 | 本轮核对结果 | 证据范围 |
|---|---|---|
| 官方是否发布处理版 | 已确认 | README与明确Zenodo record |
| 论文是否介绍预处理 | 已确认 | arXiv v7附录A.1 |
| 四类预处理源码是否可得 | 已确认 | 本地四脚本、官方同名脚本及教程 |
| 当前本地源码版本 | 已确认 | HEAD及8文件LF归一化完全一致 |
| 当前SDU档案官方校验值 | 匹配v10、v11、v12 | 本地完整档案MD5与官方清单 |
| 用户当时下载的唯一record | 尚未唯一定位 | 多record复用同一子档案 |
| 每个下载PKL的确切生成commit与运行 | 尚未逐文件认证 | 需生成manifest、依赖版本或逐值回连 |
| 处理版循环号到仪器圈号及替换来源 | 按数据集和字段认证 | 代码规则、删除/替换记录与原始记录映射 |

**处理版数据可按其发布定义与处理版坐标使用，公开代码也为进一步审计提供了直接依据。** 对真实循环距离、绝对电荷零点、时间动力学、诊断替代及早期窗口因果隔离的具体判断，需要继续使用对应的字段与来源证据。源码可得性、发布档案身份和逐文件生成关系各有可核验的证据对象。

本次来源补充属于完成实验后的只读审计说明。既有DATA-FREEZE及正式运行注册继续保留原SHA，正式科学结果及重复失配状态由其封存回执记录。

## 优势、风险与置信度

优势是用户提供的下载线索已与官方发布链对接，四类处理逻辑及版本位置可直接复查。SDU诊断替代、Stanford插值与形成圈处理、MATR续测与平移、HUST容量构造都有对应代码位置。

风险来自发布版本持续更新、未固定commit的README检索缓存差异、处理时依赖的BatteryML安装状态，以及逐文件构建记录的留存情况。论文基准队列与后续扩展数据的统计范围也需要登记。

官方处理版身份、预处理源码可得性及本地8文件版本核对的置信度为**高**。本地SDU完整档案的官方校验匹配置信度为**高**。用户当时唯一下载record与所有PKL的确切生成commit当前保持**未认证**。

## 拓展思考

处理脚本已经提供可追溯的变换定义。将原始圈索引、删除原因、诊断替换来源与处理版索引一并保存，可以直接支持物理特征的来源解释。

## 你可能想知道

同一个SDU档案在多个Zenodo版本中复用，记录号变化与子数据集字节变化之间需要分别核对。该事实也说明版本判断应结合record、档案校验值和实际文件映射。
