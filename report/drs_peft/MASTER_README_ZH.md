# DRS-PEFT 电池少样本研究阶段总索引

当前快照记录于2026-11-17。执行合同与数据、软件身份已预注册，提交为`16b37da3ea990e3d47634965b131ce8845dbc027`。

**P0与P1源开发Gate为`PASSED`，P1支持范围为窄共享梯度捕获。P2为`NEGATIVE`，P3/P4/P5为`NOT_RUN_NEGATIVE_P2_GATE`。H2/H3未评估，全部最终target Query尚无本轮正式评分。**

## 研究范围与当前状态

任务输入为Query电池前100个已认证原始循环或明确声明的供应循环VIQ、预测前已知工况，以及K颗独立Support电池的全部真实可观察未来轨迹。预测固定101–5000循环网格的SOH，评分仅使用真实可观察位置。主指标先按Query物理电芯内部观测点平均，再对电芯等权平均。

P0 pilot矩阵为2条Bridge各运行B0预训练冻结、B1随机冻结、B2标准LoRA，共6个训练单元，seed42，每单元60epochs。source85、batch8对应每epoch11updates，pilot共3960updates。source-meta平均B0/B2选桥分数为native 0.03188486638807642、embedding 0.03398819779741197，按预注册规则锁定native_multivariate。正式12单元覆盖B0/B1/B2/B3和seed42/123/2026，每单元120epochs、1320updates，正式共15840updates。两部分真实源训练合计19800updates，18份本机独立重放均通过，最大预测绝对差9.5367431640625e-7。

正式三seed的source-meta MAE均值为B0预训练冻结0.02884626、B1随机冻结0.03113055、B2标准LoRA 0.02727014、B3电池专用模型0.02704482。B0−B1配对差为−0.00228428941，seed+cell重抽样95%区间为[−0.00418150803,−0.000127554605]，seed+protocol-cluster区间为[−0.00444973678,−0.000857281531]。该证据来自源开发集及登记120epoch预算，预训练支持状态记录为SUPPORTED。

P1三seed各完成345次梯度前后向，合计1035次、optimizer updates为0。主rank4的grad_cov对random平均捕获差0.87151004，protocol区间[0.83201826,0.90801941]。对weight-SVD差0.87046484，区间[0.83235628,0.90694765]。基重合均值0.93748262。真实标签相对置换标签差−0.00878808，区间[−0.03339181,0.01574839]，结局特异性保持未建立。

P2搜索18/18与正式66/66均完整，分别完成11880和87120次更新，合计99000。gradient-core相对LoRA-XS的rank4与core56平均相对MAE改善为0.5812%与0.8510%，两个预算的cell/protocol区间均跨0，seed方向混合。相对random-core平均改善为2.4990%与3.4464%，cell区间均跨0，参数匹配预算的protocol区间为正。原预注册门槛判为NEGATIVE，P3/P4/P5依Gate保留未运行状态。

本轮真实源训练共102个单元（P0为18、P2为84），合计118800次optimizer update。P1的1035次梯度前后向与0update单独记账，工程验证另列。最终工程套件61项测试通过，耗时59.94秒。

| 阶段 | 状态 | 当前证据与下一项验收 |
|---|---|---|
| 数据、身份与元数据 | READY | 三域缓存资格、SHA与独立Support/Query分母已冻结 |
| 官方TSFM与Bridge | SOURCE_BRIDGE_FROZEN | 官方权重与两条Bridge已核验，source-meta pilot已锁native_multivariate |
| P0基线 | PASSED | pilot6及formal12均完整，源训练19800updates，最终目标Query保持sealed |
| P1退化梯度审计 | PASSED | 三seed1035次真实autograd、0update，主12QV及次42家族覆盖、梯度端点重放与loss/MAE闭合 |
| P2低维子空间对照 | NEGATIVE | 搜索18与正式66完整、99000updates，未达到两预算主控制门槛 |
| P3双视角门控 | NOT_RUN_NEGATIVE_P2_GATE | 原P2 Gate关闭后续大矩阵放行，H2未评估 |
| P4少样本核心校准 | NOT_RUN_NEGATIVE_P2_GATE | 原P2 Gate关闭后续大矩阵放行，H3未评估 |
| P5外部评测与完整交付 | NOT_RUN_NEGATIVE_P2_GATE | 最终target Query保持未评分，保留注册身份与历史暴露记录 |

## 已有文件入口

- 原始执行合同及P0–P5验收约束见[执行合同](../../research/drs_peft/EXECUTION_CONTRACT_ZH.md)。
- 必读材料存在性、完整SHA、本机依赖、历史暴露与入口见[仓库就绪审计](../../research/drs_peft/REPOSITORY_READINESS_ZH.md)。
- 全量输入、单位、mask、Stanford供应坐标与来源缺口见[数据资格报告](DATA_READINESS_ZH.md)。
- 源85/meta30/target54角色、全部目标UID和sealed用途见[数据划分锁](../../research/drs_peft/DATA_SPLIT_LOCK.json)。
- 三域目标Support/Query及5次draw见[主Episode](../../configs/drs_peft/episodes_v1.json)，source-meta Query15/Support15开发集见[Meta Episode](../../configs/drs_peft/meta_episodes_v1.json)。
- 官方权重、许可证、实际参数与子层复用见[TSFM身份审计](../../research/drs_peft/TSFM_BACKBONE_AUDIT.json)，canonical输入和两条Bridge语义见[Bridge合同](../../research/drs_peft/BRIDGE_CONTRACT.json)。
- 方法先例、LoRA-XS/LoRA-GA实现区别、LoRA-SB等直接先行工作见[方法与创新审计](../../research/drs_peft/METHOD_AND_NOVELTY_AUDIT_ZH.md)。
- 每阶段预算、方法矩阵、学习率搜索与统计设置见[执行配置](../../configs/drs_peft/execution_v1.json)。
- 当前18单元真实成绩、配对区间及执行闭环见[P0阶段审计](P0_REPORT_ZH.md)，完整CSV见[源开发汇总](../../results/drs_peft/P0/P0_SUMMARY.csv)与[预训练配对重抽样](../../results/drs_peft/P0/P0_PAIRED_PRETRAINING_BOOTSTRAP.csv)。
- 三seed真实捕获、label-shuffle边界、mean-SVD/去均值控制及432个描述性关联见[P1阶段审计](P1_GRADIENT_AUDIT_ZH.md)，原始判定见[P1 Gate](../../results/drs_peft/P1/P1_GATE.json)。
- P2完整84个单元、两预算配对区间、全部rank参数误差曲线及3颗源验证反例见[P2阶段结果](P2_RESULTS_ZH.md)，原判定见[P2 Gate](../../results/drs_peft/P2/P2_GATE.json)。

## 数据分母与测量坐标

| 域 | 独立电芯 | 合格前100曲线 | 未来有效SOH点 | 放电窗口 | 坐标 |
|---|---:|---:|---:|---|---|
| MATR | 169 | 16857/16900 | 123431 | 2.8–3.3V、4C | original_instrument_cycle |
| Stanford_2 | 181 | 18092/18100 | 170397 | 3.4–4.1V、约0.75C | supplied_cycle_number |
| HUST | 77 | 7698/7700 | 138422 | 2.8–3.1V、实测约2C末段 | original_instrument_cycle |

MATR按exact protocol group分为source_train85颗/47组、source_meta_validation30颗/15组、target_test54颗/15组，三角色的物理UID和工况组互斥。source-meta再固定每组1颗Query，共15颗，其余15颗形成pooled Support候选池，用于校准预算开发。

| 协议A的K | Q2可行目标工况组 | 固定Query电芯 | 登记draw | 各任务内unique Support集合合计 |
|---:|---:|---:|---:|---:|
| 0 | 9 | 18 | 45 | 9 |
| 1 | 6 | 12 | 30 | 21 |
| 3 | 6 | 12 | 30 | 20 |
| 5 | 4 | 8 | 20 | 13 |
| 10 | 0 | 0 | 0 | 0 |

协议A主K曲线固定4个可5-shot组的共同8颗Query，覆盖K0/1/3/5，全部可行Episode另行保留分母。6个singleton目标组缺Q2支持。exact-group K10记录`NOT_FEASIBLE`。source-meta pooled K10的Query15/Support15开发协议单独登记。

跨域Stanford固定Query91/Support90，HUST固定Query39/Support38，两者均支持K0/1/3/5/10。每K登记5draw，重复Support集合带重复标记且复用适配checkpoint。K0只有唯一空Support集合。

## 源训练、校准开发与最终目标用途

- source85提供源训练监督、source-only归一化统计及退化子空间估计。
- source-meta30提供Bridge/选权及校准学习率、步数、ridge和收缩预算开发。meta Query15与Support15物理UID互斥。
- 最终target Support未来监督仅用于方法和预算锁定后的注册适配。K0复用相同source权重。
- 最终target Query未来监督只进入锁定模型的最终评分。前缀计算、归一化、门控、源训练和选模保持角色隔离。

缓存包含全域future数组，sealed标记表示软件用途限制与评估协议。队列已有历史探索暴露，结果属于retrospective研究。模型输入allowlist保留前100圈与已知工况，单cell末次实测周期、未来mask和EOL结局留在监督及评分路径。

输入canonical x依次保留真实V、带符号C-rate、容量SOH和真实早期参考ΔQ的SOH单位。source统计拟合与物理标准差下限单独记录，全缺失工况字段保持mask。三域按各自合法电压窗编码，温箱设计、形成温度和实测温度来源分别保留。

## 计算与文件保存

本地主仓库保存代码、冻结配置、完整实验主文件、日志、选权、预测和阶段报告。服务器承担计算，结果回同步本地后复核文件SHA，再用于阶段汇总。原始数据保持只读，旧研究文件和冻结配置保留原字节。

主执行代理同期硬件实测为共享服务器8×RTX A6000，每卡49140MiB，当前指定GPU5单作业。隔离环境使用Torch2.4.1+cu121及本地vendor uni2ts。实际服务器环境、代码SHA和训练命令随ENV与运行manifest保存。本机GPU为RTX4060 Laptop，Torch2.4.1+cu121。公共产物保留硬件与软件身份，连接凭据留在私有运行环境。

B1跨OS同seed初始化出现了浮点差异，服务器实际随机backbone三seed状态已保存在本地，文件SHA与初始tensor SHA由RANDOM_BACKBONE_MANIFEST绑定。独立本机重放加载该实际状态，全部18单元验证通过。复现时同时保留随机权重与manifest，工程细节见ENGINEERING_VALIDATION_ZH.md。

P1 source-only梯度与跨平台随机控制的后续预注册提交为`04b6f79dda51e3472a40db33f1fc2fc01fea2b59`。

P0完整源开发交付提交为`bfa5a5e8e16e00e0be98a5597bc0b81c97aeed1e`，P2参数匹配与可恢复源搜索锁定提交为`a0f651f5b80c7f817a0b1ddcc48b0b0d340eb9f3`。服务器连接失败后，P1与当前P2搜索转用本机RTX4060 Laptop，模型与数据合同继续固定。

## 方案优势与劣势（风险）

- 优势为独立物理UID预算、固定Query与嵌套Support、source-only归一化、共同Head和明确Bridge身份，可逐项审计退化方向、条件化门控及目标校准的贡献。
- 劣势（风险）包括样本少且exact协议重复不均、目标高shot支持不足、历史探索暴露、Stanford BEEP插值/重编号与版本缺口，以及跨域化学体系和放电电压窗差异。
- 通用TSFM预训练是否适配电池早期VIQ需要B0/B1真实对照。P0参数、误差、耗时及显存均从真实训练凭据汇总，当前尚待结果。
- HUST名义EOL80为0事件，Stanford供应阈值和原论文寿命参考有定义差异。SOH轨迹主指标保留测量口径，EOL评估按事件与删失资格单独审查。

## 创新边界与假设

`W0 + U C Vᵀ`、权重SVD小核心和梯度导出固定低维基均已有先行工作。LoRA-XS、LoRA-GA、LoRA-SB等实现与公开原始定义的具体差异已登记。研究增量需要分别证明跨源工况退化梯度的稳定性、绝对/相对早期响应的条件化价值及目标少样本核心校准收益。

| 假设 | 当前判断 | 所需证据 |
|---|---|---|
| H1共享退化梯度子空间 | 共享捕获支持，退化特异性待验证 | 三seedrandom/W-SVD捕获及稳定性通过，label-shuffle控制未显示弱化 |
| H2双视角条件化 | 未评估 | P3按NEGATIVE_P2_GATE保留未运行状态 |
| H3少样本核心校准 | 未评估 | P4按NEGATIVE_P2_GATE保留未运行状态 |

带载响应分解反映已观测电学结构，电极级LLI/LAM/SEI真值及微观机理解释需要额外认证。轨迹低维性到参数梯度低维性的关系由P1实测审查。

## 阶段Gate与负结果处理

| 阶段 | 进入下一步的验收条件 | 负结果处理 |
|---|---|---|
| P0 | 官方权重真实加载、Bridge语义、真实训练/推理闭环与泄漏QA通过 | 预训练无增益时保留源验证差异和Bridge诊断，登记风险后继续P1 |
| P1 | 留出源工况上可重复优于random/W-SVD的梯度捕获或更新收益 | 记H1_NOT_SUPPORTED，可继续注册P2对照研究，按合同暂停P3/P4大矩阵 |
| P2 | 源开发及独立源组留出中gradient-core相对LoRA-XS/random具有证据 | 保留负结果，转换研究假设需注册新版本 |
| P3 | TWO_VIEW相对P2/STATIC的增量及shuffle/质量控制符合预期 | 缺乏增量时保留P2结构进入P4 |
| P4 | K>0增量可归因于注册Support，且与LoRA/LoRA-XS公平比较 | 保留负迁移、各K/各draw/各域负结果，按证据限定H3归因 |
| P5 | 方法与超参数冻结、最终Query评分、独立重放和统计重算全部完成 | 所有失败电芯、外推例和缺失分母进入报告 |

继续/停止依照预注册源验证证据。最终Query评分留到方法与校准预算全部锁定后。`SOURCE_DEVELOPMENT_COMPLETE.json`对应源开发完成，最终`COMPLETED.json`还要求正式测试与重放验收。

## 复现入口

以下PowerShell入口在仓库根目录执行。`DRS_BACKBONE_DIR`需指向已核验官方Moirai权重缓存，服务器运行使用其隔离环境Python和同SHA代码。

```powershell
& D:\Anaconda\envs\tslib\python.exe -m scripts.drs_peft.readiness
& D:\Anaconda\envs\tslib\python.exe -m scripts.drs_peft.episodes
& D:\Anaconda\envs\tslib\python.exe -m scripts.drs_peft.train --help
& D:\Anaconda\envs\tslib\python.exe -m scripts.drs_peft.summarize_p0
& D:\Anaconda\envs\tslib\python.exe -m scripts.drs_peft.summarize_p1
& D:\Anaconda\envs\tslib\python.exe -m scripts.drs_peft.summarize_p2
```

P0 pilot的等价入口如下，登记矩阵固定为2个Bridge×3个方法×seed42×60epochs。服务器实际命令及环境以运行manifest为准。

```powershell
& D:\Anaconda\envs\tslib\python.exe -m scripts.drs_peft.train `
  --config configs/drs_peft/execution_v1.json `
  --data-root runtime/drs_peft/data_v1 `
  --output-root results/drs_peft `
  --backbone-dir "$env:DRS_BACKBONE_DIR" `
  --phase pilot --device cuda --resume
```

pilot完整结果仅用source-meta选择Bridge。之后同一入口的`--phase formal`使用锁定Bridge运行B0/B1/B2/B3、配置中的3个seed和120epochs。P1正式入口为`scripts.drs_peft.run_p1`，须具备P0 Gate及各seed匹配的正式B0选权。后续命令随实际实现与阶段结果登记。

## 后续交付清单

以下列出原合同交付覆盖。P0/P1/P2的source阶段产物已实际形成，P3/P4/P5依据阴性门槛记录未运行。

| 阶段 | 必需文件和证据 |
|---|---|
| P0 | P0_REPORT_ZH.md、P0_METRICS.csv、P0_FAILURES.json、DATA_SPLIT_LOCK.json、TSFM_BACKBONE_AUDIT.json、BRIDGE_CONTRACT.json，以及B0/B1/B2/B3真实训练、参数拆分、权重SHA、固定验证预测及独立重构 |
| P1 | P1_GRADIENT_AUDIT_ZH.md、逐层CSV、主角度矩阵、random与W-SVD对照、SVD图、配置锁、源UID和状态指纹、H1 Gate |
| P2 | P2_RESULTS_ZH.md、P2_METRICS.csv、P2_PARAM_BUDGET.csv、同rank与匹配可训练参数表、多seed结果、模型检查、逐cell预测与反例 |
| P3 | NOT_RUN_NEGATIVE_P2_GATE，双视角和局部消融性能未评估 |
| P4 | NOT_RUN_NEGATIVE_P2_GATE，目标K-shot校准与负迁移性能未评估，Episode合同已冻结 |
| P5 | NOT_RUN_NEGATIVE_P2_GATE，外域最终Query性能未评分，创新与数据边界保留在source报告 |

每个正式run还需RUN_CONFIG、SOURCE_SNAPSHOT、DATA_FINGERPRINT、EPISODE_IDS、BACKBONE_IDENTITY、TRAINABLE_PARAMETER_REPORT、TRAIN_LOG、CHECKPOINT_SELECTED、CHECKPOINT_SELECTION、TEST_PREDICTIONS、TEST_CELL_METRICS、RESULT_METRICS及最终COMPLETED凭据。各阶段保存真实命令、有效分母、源/目标可训练参数、优化更新、GPU实耗和峰值显存。

最终结果覆盖DATA_READINESS、TSFM_BASELINE、GRADIENT_SUBSPACE、PEFT_COMPARISON、FEWSHOT_SCALING、ABLATION、COMPUTE、FAILURES八类事实表。物理电芯配对与工况组敏感性统计按注册预算计算，训练seed、Support draw与Query电芯层级分别呈现。

## 置信度与后续关注

数据计数、SHA、角色、完整训练、重放及P2阴性门槛的置信度为高，均有实际产物支撑。P1支持固定source checkpoint上的共享低维梯度捕获，退化特异性未建立。H2/H3没有对应实验证据。source-meta复用与既有历史探索暴露限定结果的解释范围。

本轮结束于原合同的P2阴性门槛，完整平均值、区间、混合seed方向和高误差个体均保留。P1 mean-SVD捕获差具有seed差异，全局mean-SVD/demeaned_cov的专属bootstrap稳定性及mean-SVD基完整下游训练仍有缺口，相对LoRA-SB的优越性仍需对应忠实对照。共同8Query及全部Support/Query身份继续冻结，未来研究版本保留当前未评分状态与历史暴露前史。
