# DRS-PEFT 先行工作与可允许论述范围

## 审计范围与证据状态

检索日期为 2026-11-17。本表对应合同 §8 的先行工作审计。检索发起时，正式 P0 正在运行，P1/P2 尚无正式实验结果。表中的方法区别属于文献与协议比较，其增益仍需对应阶段的实际产物验证。

**梯度来源的冻结子空间加可训练小核心已有直接先行工作 LoRA-SB。Timer 加 LoRA 的电池容量预测也已有公开论文。** 这些事实限定本轮后续创新论述的范围。[LoRA-SB 论文](https://arxiv.org/html/2411.19557v4)、[Battery-Timer 最新预印本](https://arxiv.org/html/2505.08151v4)

本表将数值序列上的广泛预训练骨干及其复用记为“通用 TSFM”，将多电池数据预训练的专用模型记为“电池领域 FM”。各条目的实际任务、输入和监督预算同时列出。`UNVERIFIED` 表示本次查看的原始证据尚不足以核实该字段。

## 身份、任务与数据预算

| 条目与年份 | 论文及正式标识 | 预训练对象与实际任务 | 数据与监督预算 | 原始来源 |
| --- | --- | --- | --- | --- |
| LoRA，2021 首稿，ICLR 2022 | Low-Rank Adaptation of Large Language Models，arXiv 2106.09685 | NLP 语言模型适配，包含 RoBERTa、DeBERTa、GPT 系列 | GLUE 与文本生成等任务的监督数据，预算按 NLP 样本计 | [论文](https://arxiv.org/abs/2106.09685)、[官方代码](https://github.com/microsoft/LoRA) |
| LoRA-XS，2024 首稿，ECAI 2025 | Low-Rank Adaptation with Extremely Small Number of Parameters，arXiv 2405.17604，DOI 10.3233/FAIA251185 | NLP 语言模型适配 | GLUE、170K 常识样本、100K MetaMathQA；完整训练数据与 SVD 初始化分别记账。出版社页面在线时间为 2026-08-25，卷名为 ECAI 2025 | [v3 全文](https://arxiv.org/html/2405.17604v3)、[出版社](https://journals.sagepub.com/doi/10.3233/FAIA251185)、[官方代码](https://github.com/MohammadrezaBanaei/LoRA-XS) |
| LoRA-SB，2024 首稿，2025 v4 | Initialization using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning，arXiv 2411.19557 | NLP 语言模型适配 | 50K MetaMathQA、CommonSense170K、GLUE。初始更新估计使用各训练集的 0.1%，随后仍使用相应完整监督训练预算 | [v4 全文](https://arxiv.org/html/2411.19557v4)、[官方代码](https://github.com/CERT-Lab/lora-sb) |
| LoRA-GA，2024 | Low-Rank Adaptation with Gradient Approximation，arXiv 2407.05000 | NLP 语言模型适配 | GLUE；Llama2-7B 实验分别使用 52K WizardLM、100K MetaMathQA、100K Code-Feedback。梯度估计是训练前的额外数据使用步骤 | [v2 全文](https://arxiv.org/html/2407.05000v2)、[官方代码](https://github.com/Outsider565/LoRA-GA) |
| HiRA，ICLR 2025 | Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models，OpenReview TwJrTz9cRS | NLP 语言模型适配 | 常识推理、对话、数学任务。官方示例使用 `common_170k`、rank32、3 epochs。电芯 K-shot 预算未定义 | [官方会议页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/48c368f105e8145b945227b73255635a-Abstract-Conference.html)、[官方代码](https://github.com/hqsiswiliam/hira) |
| ABBA-Adapters，2025 首稿，ICLR 2026 | Efficient and Expressive Fine-Tuning of Foundation Models，arXiv 2505.14238；v1 题名为 Highly Expressive Hadamard Product Adaptation for Large Language Models | NLP 语言模型适配 | CommonSense170K、20K MetaMathQA；论文按实际参数量匹配 LoRA 与 ABBA | [v4 全文](https://arxiv.org/html/2505.14238v4)、[官方代码](https://github.com/CERT-Lab/abba) |
| Time-PEFT，ICML 2026 | Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models，PMLR 306:91489–91508 | **通用 TSFM**，官方论文使用 MOMENT-base，处理多变量时间序列预测 | 按目标时间序列训练窗口监督。已核代码包含 ETT 固定时间段划分及自定义数据的 70/10/20 时间划分；论文全部实验训练比例与窗口数为 `UNVERIFIED` | [官方论文页](https://proceedings.mlr.press/v306/na26b.html)、[官方代码](https://github.com/kaist-dmlab/TimePEFT)、[数据划分代码](https://raw.githubusercontent.com/kaist-dmlab/TimePEFT/main/data_provider/data_loader.py) |
| Battery-Timer，2025 首稿，RESS 2026 | Foundation Models Knowledge Distillation For Battery Capacity Degradation Forecast，arXiv 2505.08151，DOI 10.1016/j.ress.2025.112139 | **通用 TSFM Timer-84M**，容量历史到未来容量，并向小型专家蒸馏 | 最新 v4 使用 220,153 次循环记录微调，5 epochs。SJTUIE 的 8 颗电池用于验证，窗口为过去96点预测未来96点 | [最新 v4](https://arxiv.org/html/2505.08151v4)、[期刊页](https://www.sciencedirect.com/science/article/pii/S0951832025013389)、[官方代码](https://github.com/sjtu-chan-joey/Battery-Timer) |
| BatteryMFormer，KDD 2026 | Multi-level Learning for Battery Degradation Trajectory Forecasting，arXiv 2605.27044，DOI 10.1145/3770855.3818948 | 电池专用监督 Transformer，包含时间/SOC 双视角、老化条件解码和轨迹记忆；条件文本使用 Qwen3 embedding | 前至多100圈、圈内长度300，预测 SOH 轨迹至5000。四域分别963/27/31/95颗电池，另有保留50%训练电池的实验 | [v2 全文](https://arxiv.org/html/2605.27044v2)、[正式 DOI](https://doi.org/10.1145/3770855.3818948)、[官方代码](https://github.com/Ruifeng-Tan/BatteryMFormer) |
| PBT，2025 首稿，EES 2026 | Pretrained battery transformer，arXiv 2512.16334，DOI 10.1039/d6ee03446b | **电池领域 FM**，预训练一般 PBT 后构建场景专用模型，输出电池循环寿命标量 | 13 个锂离子数据集预训练，15 个下游数据集含977颗电池与532种老化条件。官方接口支持前 N≤100 圈，精确目标 K/比例为 `UNVERIFIED` | [最新 v7 摘要](https://arxiv.org/abs/2512.16334v7)、[正式 DOI](https://doi.org/10.1039/d6ee03446b)、[官方代码](https://github.com/Ruifeng-Tan/PBT) |

这里的 PBT 固定指 Ruifeng Tan 等人的 Pretrained Battery Transformer，ABBA 固定指 CERT-Lab 的 ABBA-Adapters，HiRA 固定指 Hadamard High-Rank Adaptation。上述身份已由题名、作者项目和原始论文相互核对。BatteryMFormer 的原文给出监督训练流程，其预训练文本条件模块有独立说明，所提电池模型的通用 TSFM checkpoint 来源在本次证据中为 `UNVERIFIED`。[PBT 官方项目](https://github.com/Ruifeng-Tan/PBT)、[ABBA 官方项目](https://github.com/CERT-Lab/abba)、[HiRA 官方会议页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/48c368f105e8145b945227b73255635a-Abstract-Conference.html)、[BatteryMFormer 方法与训练部分](https://arxiv.org/html/2605.27044v2)

## 适配参数、已存在的内容与本轮差异

以下 `m×n` 为一层权重大小，参数量均按对应适配层计算。整模型还应累计所有目标层、Bridge、Head、物理支路和可训练条件模块。

| 方法 | 可训练对象与适配预算 | 与本轮重叠 | 本轮仍需证明的差异 |
| --- | --- | --- | --- |
| LoRA | `ΔW=sBA`，每层 `r(m+n)` | 冻结 W0、低秩更新 | 相同 Bridge/Head 和验证预算下的结构化方法增量 |
| LoRA-XS | 冻结 W0 的 SVD 衍生基，包括奇异值尺度，训练 `r×r` 核，每层 `r²` | `U C Vᵀ` 小核心与极低目标参数预算 | 源电池退化方向的来源价值及跨域稳定性 |
| **LoRA-SB** | 初始 full-FT 更新或梯度近似决定冻结基，训练 `r×r` 核，每层 `r²` | **梯度衍生固定基、小核心、投影更新均直接重叠** | 跨源电池/协议学习的共享方向，以及目标 K 颗完整电池下的校准与负迁移控制 |
| LoRA-GA | 梯度对齐初始化 A/B，并进行初始函数补偿；随后训练 A/B，每层 `r(m+n)` | 梯度方向用于适配初始化 | 固定子空间核心与可移动 A/B 在真实预算下的表现差异 |
| HiRA | `ΔW=W0⊙(BA)`，A/B 可训练，每层 `r(m+n)` | 结构化乘性调制与参数效率 | 本轮逐电池物理条件门控的独立贡献，及其目标更新限制 |
| ABBA | `ΔW=s(B1A1)⊙(B2A2)`，每层 `(r1+r2)(m+n)` | Hadamard 乘法与结构化更新 | 固定退化子空间、可训练核心和逐电池条件化所形成的可证增量 |
| Time-PEFT | 当前官方代码训练 encoder LoRA、frequency adapter、channel adapter 和 forecast head；W0 冻结。论文完整总参数为 `UNVERIFIED` | TSFM 的数据结构感知 PEFT、额外输入适配器 | 电池退化信息如何限定参数方向，以及以完整电芯计数的 K-shot 泛化 |
| Battery-Timer | 最新 v4 注入 q/k/v，rank8、alpha16、dropout0.05；报告 LoRA 为总参数的0.4673% | 电池 TSFM、LoRA、退化先验和跨条件迁移 | 早期 VIQ→完整 SOH 轨迹、源域共享子空间和冻结其余模块的目标小核心校准 |
| BatteryMFormer | 电池模型主体、双视角、条件解码和记忆参与监督训练。精确可训练总数为 `UNVERIFIED` | 局部电化学响应、双视角与条件化早期轨迹预测 | 这些信息进入 TSFM 适配矩阵的具体作用，以及同信息预算下的少样本收益 |
| PBT | 官方提供整模型 FT 以及 AT/AT_nCP adapter tuning；官方脚本暴露 adapter_size16。精确 AT 参数总数为 `UNVERIFIED` | 多电池预训练、低数据目标专用化和 adapter 迁移 | 完整轨迹任务上的同信息预算比较，以及目标只更新 δC 的校准行为 |

前三类小核心与梯度方法的公式和参数范围分别由论文与作者实现核对。LoRA-XS 的 Σ 放置还需结合代码的行/列权重约定检查，本仓库对应检查见 `METHOD_AND_NOVELTY_AUDIT_ZH.md`。LoRA-SB 在当前冻结 P2 方法列表之外，其直接先行地位仍需完整保留在论述中。[LoRA 原文](https://arxiv.org/abs/2106.09685)、[LoRA-XS 原文](https://arxiv.org/html/2405.17604v3)、[LoRA-SB 原文](https://arxiv.org/html/2411.19557v4)、[LoRA-GA 原文](https://arxiv.org/html/2407.05000v2)

HiRA 的 Hadamard 因子包括冻结 W0，ABBA 则有两对独立可训练因子。它们的已学习更新在推理阶段具有固定参数。本轮合同 P3 的门控由每颗电池的早期绝对响应与相对漂移生成，这一算子差异可以准确描述，其效果需要 STATIC_GATE、打乱漂移和预算匹配对照。[HiRA 官方说明](https://github.com/hqsiswiliam/hira)、[ABBA 方法公式](https://arxiv.org/html/2505.14238v4)

Time-PEFT 当前官方 `run.py` 的训练范围包含 LoRA 和两个 adapter 及 Head。依据该代码静态计数，frequency projection 有 `D²+D` 个参数，channel adapter 有 `2Dh+h+C(hD+D)+2D` 个参数，其中 `h=D//hidden_dim_ratio`、C 为通道数。还需加所有 LoRA 与 Head 参数。该推导描述本次查看的公开代码，论文全部配置的总数继续记为 `UNVERIFIED`。[官方训练与模块代码](https://raw.githubusercontent.com/kaist-dmlab/TimePEFT/main/run.py)

Battery-Timer 存在版本变化。v3 报告6类投影、1.3068%和480→196验证窗口，最新 v4 报告 q/k/v、0.4673%及96→96窗口。上表采用 v4。PBT 的官方 transfer 脚本默认 FT，README 同时说明 AT 与 AT_nCP，参数报告应绑定具体模式。[Battery-Timer v3](https://arxiv.org/html/2505.08151v3)、[Battery-Timer v4](https://arxiv.org/html/2505.08151v4)、[PBT transfer 说明](https://github.com/Ruifeng-Tan/PBT)、[PBT transfer 脚本](https://raw.githubusercontent.com/Ruifeng-Tan/PBT/main/scripts/finetune_PBT.sh)

## Timer 电池先行工作的补充覆盖

另核实到 *Universal Battery Capacity Degradation Forecasting Driven by Foundation Models Across Diverse Chemistries and Conditions*，Advanced Science，2026-08-24 首次在线，DOI `10.1002/advs.77261`，对应近题名预印本 arXiv `2601.00862` 于2025-12-30提交。论文描述20个数据集、1704颗电池、3,961,195个循环片段，采用 Timer、LoRA 和物理描述子对比学习，容量历史用于预测，物理视角参与训练。该工作进一步覆盖“电池 TSFM+物理信息+参数高效迁移”的宽泛组合。精确 LoRA rank、目标层、总参数与严格电芯 K-shot 预算在本次查到的正文中为 `UNVERIFIED`，方法细节指向补充材料。[出版社原始条目](https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/advs.77261)、[预印本全文](https://arxiv.org/html/2601.00862v1)

## 本轮可检验的研究问题

| 待检验内容 | 已有重叠 | 对应本轮证据要求 | 当前证据状态 |
| --- | --- | --- | --- |
| 源域退化方向具有跨电池、跨协议的共享性 | LoRA-SB/GA 已利用梯度方向 | P1 的源组留出、随机/权重SVD对照及稳定性；P2 的匹配 rank/实际参数量/搜索预算结果 | 待正式 P1/P2 |
| 电池物理双视角能够控制适配方向 | BatteryMFormer 有局部与双视角，Timer 相关工作已有物理训练信息 | P3 的无门控、单视角、静态、打乱、去局部及预算控制 | 待正式 P3 |
| K 颗目标电池足以校准共享核心 | PBT 已有低数据专用化与 adapter 迁移 | P4 的实体/协议隔离、同一 Query 组成、真实独立 Support 集计数、只更新 δC、负迁移逐域报告 | 待正式 P4 |
| 通用 TSFM 的预训练知识在该任务上有价值 | Time-PEFT/Battery-Timer 已适配通用 TSFM | P0 的真实预训练/随机冻结/LoRA/电池模型对照，共享任务头、信息预算和公平选权 | P0 运行中，结论待完成产物 |

因此，当前可写为“研究源域电池退化方向在固定 TSFM 子空间与目标小核心校准中的作用”。“提出 UCVᵀ”“首次梯度子空间 PEFT”“首次电池 TSFM+LoRA”“首个少数据电池迁移模型”等表述已有直接覆盖。候选方法的整体全球首创性仍为 `UNVERIFIED`。[LoRA-SB](https://arxiv.org/html/2411.19557v4)、[Battery-Timer](https://arxiv.org/html/2505.08151v4)、[PBT](https://arxiv.org/abs/2512.16334v7)

## 检索边界、优势与风险

检索采用精确方法名、完整题名和相关组合词，包括 `Time-PEFT`、`Timer LoRA battery`、`BatteryMFormer`、`PBT battery transformer`、`LoRA-SB`、`HiRA Hadamard` 与 `ABBA adapters`。直接尝试 Google Search 的 Time-PEFT 与 Timer-LoRA 查询页均返回工具不可访问，随后使用可用 web 搜索，并打开 arXiv、会议/出版社页面及作者 GitHub 交叉核实。结果属于定向先行审计。

优势在于主要结论有原始方法公式、版本历史和官方代码支持，监督预算同时记录数据量与样本单位。风险包括以下仍待核实字段。

- Time-PEFT 的官方 PMLR PDF 被网页工具判为不支持的 octet-stream，OpenReview PDF 返回浏览器验证。本表使用官方论文页与公开源码，完整论文预算、独立 arXiv 标识及 DOI 为 `UNVERIFIED`。
- PBT 的最新 v7 摘要、正式 DOI 与官方 transfer 入口已核实，全文 HTML 和出版社正文在本次入口不可访问。精确目标 K/训练比例、各 adapter 模式总参数为 `UNVERIFIED`。
- BatteryMFormer 的总体数据量与50%训练集实验已核实，各域最终训练电池绝对数量和模型可训练总参数为 `UNVERIFIED`。
- 本轮与既有工作的误差数值受输入、监督单位、时间轴、目标形式、预测窗和划分影响。当前文献表支持方法边界审计，科学优劣需统一协议的正式对照。

**核心重叠判断的置信度为高**，依据为已打开的原始公式、作者实现及官方实验描述。未核实字段保持 `UNVERIFIED`，候选方法整体创新性与有效性的判断等待正式证据。

检索记录采用 paper-lookup 技能的定向检索与标识核对流程。该工具库的研究引用为 Kassis 等（2026），用于说明检索过程。[Scientific Agent Skills 工具库论文](https://arxiv.org/abs/2609.00065)
