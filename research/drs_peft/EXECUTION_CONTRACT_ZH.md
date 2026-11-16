# Codex 执行合同：DRS-PEFT 电池退化特征驱动的 TSFM 少样本参数高效微调研究

> 版本：v1.0（2026-11-17）
> 工作仓库：`fallenplum/battery`，当前代码所在本地项目优先；如果本地未找到，检查已授权的 GitHub 同名仓库。
> 任务性质：**实际审计、编码、训练、评估和交付**。不能将设计文档、随机前向检查、模拟结果写成已完成的真实实验。
> 核心假设：电池早期 VIQ 中的退化相关结构，可帮助识别和控制通用 TSFM 的有效参数更新方向；目标域极少量完整寿命标签能够通过低维受约束适配实现有效迁移。
> 研究终点：前 100 个可认证循环的 VIQ + 预测时已知工况 -> 查询电池未来**实际可观测** SOH 轨迹；目标域 K=0/1/3/5/10 颗**独立且具有完整已观测轨迹的支持电池**。严禁把同一颗电池的数百个循环当成数百个 shot。

---

## 0. 执行方式和不可妥协的科学纪律

你是科研工程负责人，不是仅写计划的助手。先检查真实仓库、数据、环境和算力，然后按 P0→P5 顺序执行；**禁止直接跳到完整 DRS-PEFT**。不要因为某阶段结果不好就无记录地修改数据集、物理定义、目标指标或测试划分。保持旧研究文件、checkpoint、DATA_FREEZE 和历史结果完全不变，全部新代码、数据合同与结果进入独立命名空间：

- `research/drs_peft/`：新研究合同、假设、风险和运行注册。
- `configs/drs_peft/`：版本固定的配置、数据资格、预设 Episode、参数预算及模型矩阵。
- `models/drs_peft/`：TSFM 适配、PEFT 方法和预测头。
- `scripts/drs_peft/`：数据接入、训练、评估、诊断及独立核验。
- `results/drs_peft/`：原始结果、每个 run 的完整配置/日志/权重/预测/状态。
- `report/drs_peft/`：中文阶段报告、主结果表、图、负例和最终总结。

若仓库已经存在上述路径，检查并采用新的实验版本，不覆盖原先交付。新任务建立 git 分支或明确的独立提交记录；涉及极大文件的 raw/checkpoint 缓存留在本地 gitignored 路径，只提交小型可复核交付和 manifest。勿自动删除任何旧数据或 checkpoint。

**禁止：** 测试标签参与训练/选模/标准化；根据测试结果改窗口、特征或超参数；用测试电池未来长度 T_i 当成推理输入；把预测头改变带来的改善宣称为 PEFT 改善；偷偷替换 TSFM checkpoint 为随机初始化；用 source/target 相同电池身份或跨批续测在 Support 和 Query 两侧；使用虚构或合成的正式性能；只跑 smoke 就称训练完成；将只在代码中实现的 P2–P5 称为已完成科学实验。

**执行规则：** 每阶段至少交付可重复命令、配置 SHA、数据指纹、有效样本数、模型权重识别码、可训练参数数目、真实训练日志、独立预测重放和实际误差。如环境、数据或 GPU 不具备条件，先完成能做的审计/测试，明确 `BLOCKED_REASON`，严禁填充结果。每阶段的结果若与假设相反，保留负结果并执行预注册的停止/分叉规则，不能自行美化。

## 1. 必读仓库材料及已知事实（先核验，不能直接假设全都可用）

优先实读如下文件：

1. `README.md`
2. `contracts/SRSR_Physics_统一研究问题_一次性数据处理与模型实验总合同.md`
3. `configs/unified_matr_seed42.json`
4. `data/unified_matr_seed42/DATA_FREEZE.json`
5. `data/unified_matr_seed42/dataset_qa_report.md`
6. `data/unified_matr_seed42/split_contract.json`
7. `models/unified_matr/README.md` 和 `models/unified_matr/model.py`
8. `scripts/unified_matr/train_seed42.py` 以及现有训练 loader、验证和独立核验入口。
9. `report/UNIFIED_MATR_SEED42_FINAL_ZH.md`
10. `report/M4_FULL_METHOD_FINAL_ZH.md`
11. `report/M4_ABLATIONS_SEED42_FINAL_ZH.md`
12. `report/MULTIFAMILY_EXTENSION_REPORT_ZH.md`
13. `report/MATR_EXTENSION_QUALIFICATION_ZH.md`
14. `report/STANFORD_EXTENSION_QUALIFICATION_ZH.md`
15. `report/HUST_RAW_IDEA_FEASIBILITY_ZH.md`
16. `report/SDU_DATA_AUDIT_ZH.md`

当前记录的基线事实（复读源文件确认后写进审计报告）：

- MATR 冻结主队列：169 颗唯一物理电芯；前100圈合格原生 Q(V) 16857/16900 条；未来有效 SOH 123431 点；名义 EOL80 事件 169/169；原生循环真实轴已认证；完整 DATA_FREEZE 与 580 项核验已通过。
- MATR 的典型接口：`raw_viq:[B,100,64,3]`、`point_mask:[B,100,64]`、`cycle_ids:[B,100]`、`reference_viq`、`global_features:[B,100,20]`、`local_residual:[B,100,64]`、`local_features:[B,100,8,26]`、policy/masks、`anchor_soh100`。真实维度以 loader/冻结数据为准。
- HUST 77 颗均有原始前缀及未来 SOH，主放电视图已冻结；名义 EOL80 事件为 0/77，不能拿 HUST 当作 EOL80 回归真值集。
- Stanford_2 当前合格供应前缀 181 颗，提供形成工艺元数据，但使用 BEEP 处理/重编号坐标，真实原生周期轴与具体转换版本尚未完成逐文件认证；不许混称原生 K100。
- SDU 86 颗当前快照存在重复循环记录及原始 protocol/循环身份缺口，不能默认直接入主矩阵。
- 历史同任务但非 Few-Shot 的 MATR：M1 主 OOF MAE ~0.018762，M2 ~0.018946，完整 M4 ~0.023271，M4-A1 ~0.020148；均属于**既有历史探索集和不同训练协议**，不可直接与本次 K-shot 主表横向宣称 SOTA。
- M4 的严格非负速率积分头在主 OOF 和未见协议上没有证明优于旧 Flow；新 PEFT 研究第一轮必须锁定**相同预测头**，将输出头创新与 PEFT 创新分离。

生成 `research/drs_peft/REPOSITORY_READINESS_ZH.md`：逐文件记录存在性、实际字段、SHA/版本、数据权限、训练代码入口、缺口、系统环境、GPU 型号/显存、Torch/CUDA/Python、依赖安装状态。保留原始结论和本轮建议之间的界线。

## 2. 唯一研究问题、三个假设及成功标准

任务：研究一种**电池退化响应结构化 PEFT**，对公开预训练的 TSFM（首选 Moirai 系列轻量权重）进行有限参数源域专门化，然后利用新目标域 K 颗完整寿命支持电池完成快速适配，预测完全不同的 Query 电池未来 SOH 曲线。

统一模型输入：单颗 Query 电池前100个原始或明确声明的供应循环的 VIQ（及可从前100圈计算的变化量）、预测时已知 policy、支持集 K 颗**具有全部真实已观测未来轨迹**的独立电池。统一模型输出：固定、预登记的未来循环查询网格上的 SOH 预测，并只在真实有标签的位置计算误差；Query 的末次实测循环数和 EOL 未来事件不可作为预测输入。

主指标：Query **cell-mean** full observed future SOH MAE；次指标 cell-mean RMSE、101–200/201–500/501–1000/1001+ 四段 MAE、工况组宏均值、名义 EOL80（仅有实际事件/删失可比性的队列）、负迁移率、可训练参数和峰值显存、GPU 实耗。必须固定每颗电池的轨迹误差先归一，再跨电池等权平均。

三个需实证检验的假设：

- **H1（参数子空间）**：来自不同源域工况的退化轨迹损失梯度中存在比随机子空间更稳定、且能改善未见工况更新方向的共享低维结构。
- **H2（双尺度调制）**：同一 PEFT 子空间内，输入电池的绝对 VIQ 响应与前缀跨循环相对变化具有互补的门控信息，在相同训练预算和匹配参数下改善 Query 预测。
- **H3（Few-Shot 核心校准）**：将目标域适配限制在冻结子空间的低维核心，可在 1/3/5-shot 下，比标准 LoRA 或强低维基线更稳定地利用完整寿命支持数据。

禁止事先声明“电池预训练先验和 TSFM 周期先验必然冲突”“梯度低维性由退化轨迹低维性直接推出”“物理规律一定改善精度”。对前两者可分别设计经验检验，理论只在明确假设下推导。

## 3. P0：一次性数据 / Baseline 接口冻结 + 预训练有效性（本阶段必须先完成）

### P0-A. 创建严格的 Few-Shot Episode 合同

定义三种互不混淆的协议：

**A. MATR 未见协议适配：** 以 verified physical cell UID 和预先登记的 exact protocol group 划分。源域训练组、源域模型选择组、最终目标测试工况互斥。在某个 held-out 目标协议组里，先为各次重复固定 Query，再从**非 Query**候选池抽 K 颗 Support；要求组内实际独立数至少 K+Q，至少保留1颗独立 Query，建议不少于2颗，否则该 Episode 跳过并显示分母。只做样本支持的 K；不能为5-shot重复抽取一颗电池。多个 source/meta-validation 的工况群必须来自不同物理 UID，事先冻结。

**B. MATR -> Stanford_2：** 源域仅使用源训练电池的未来标签；目标 Stanford 测试集提前分割固定 Query 池与 Support 候选池，同一 K-shot draw 下各方法共享支持电池，Query 固定不随 K 改变。预处理使用 Stanford 的合法共同放电窗口和供应周期坐标，给每个指标标明 `supplied_cycle_number`，不擅自称其为原生仪器周期。目标域 K=0/1/3/5/10，并将不同化学体系/额定容量的变化纳入显式元数据。

**C. MATR -> HUST：** 同理，目标 HUST 仅以完整可观察未来 SOH 做校准。主放电视图为实际2C末段窗口（与 MATR 的4C 2.8–3.3 V不共用电压 bin 直接含义）；名义 EOL80 不做无事件精度评价。K=0/1/3/5/10 可按支持池数量审查后决定。

**SDU：** 原始身份链不合格前保持候选，不加入正式 Few-Shot 主结果；可单独做支持率审计。

主研究使用 K ∈ {0,1,3,5,10}，若某个 Episode 无足够独立支持/查询，记录 `NOT_FEASIBLE`，不变换任务定义。对于每个 K 预先生成固定的支持集重复（建议5次，正式规模可根据审计后有限计算预算冻结），所有模型共享完全相同的 Support UID / Query UID。将种子和哈希写入 `configs/drs_peft/episodes_v1.json`，并单独记录模型训练 seed（建议42、123、2026；pilot 可先42）。

训练折内拟合标准化；查询数据未来截断、重排、极端篡改应不影响任何前100圈特征或条件化门控。特征选择、政策编码与窗口不能使用 Query 真实未来长度。新建 dataset adapter 需输出完整 data manifest、base/support/query UID、source version、目标坐标、容量分母、观测 mask 和日志。

### P0-B. TSFM 官方权重和接口审计

首选 Salesforce 官方 `uni2ts` 的 **Moirai-1.1-R-small**（确切可用模型标识、commit、权重 SHA、许可证及 API 按实际安装登记）；备选官方 Timer 或另一公开且可训练的轻量 TSFM。禁止使用未经验证的伪预训练 checkpoint，禁止默认从零训练 PBT 等大型模型。参考官方项目：`https://github.com/SalesforceAIResearch/uni2ts` 和 `project/moirai-1/finetune_lsf/README.md`。

必须解决**模型原生输入 vs 电池前100圈二维 VIQ**接口问题。先做两条无泄漏、可审计的输入 Bridge 候选（用源域验证集选择，不能看目标测试）：

1. **Native multivariate bridge**：每圈先由固定、可复制、严格前缀计算的电学特征组成原生时序 `[B,100,C']`；进入官方 TSFM 的多变量分词/前向路径，说明特征名称、数值单位、C'、patch 与 mask。
2. **Cycle embedding bridge**：将 `[B,100,64,3]` 经过共享的圈内编码器转为 `[B,100,d]`，接入 TSFM 的正式隐藏层路径；明确该路径是否绕过预训练 patch embedding；记录仍然真正复用的预训练子层比例及结构改造，不能将其错误称为“完整原生 TSFM”。

两条 Bridge 都要求一份示例前向测试：随机、真实电池、缺失与掩码、电芯身份及 Query 将来标签扰动、GPU/CPU、batch=1/8 维度、可微性和无未来泄漏。桥接选型只用源验证数据；各 PEFT 方法共享**同一个选定 Bridge**。输入 Bridge 和轨迹输出 Head 的可训练参数要单独报告，不能隐藏在 PEFT 参数预算之外。

统一输出头第一轮只采用一个**相同**的 cycle-query 连续轨迹头：`state_embedding + normalized query_cycle -> SOH`。预测 Query 网格预先固定，例如整数101–2500（不足观测位置在评分时 masked），另将2501–5000作为诊断范围；不以电芯真实 T_i 决定预测最大查询长度。维持 SOH100 观察锚点、同一 loss、同一 train/val 选权规则。**P0–P4 不以改变预测头来给新 PEFT 增益背书。** 

### P0-C. 最小基线（真实训练，不允许仅 smoke）

固定源开发分割，至少训练并保存：

- `B0`: Pretrained TSFM，完全冻结 backbone，仅训练相同 Bridge/Head。
- `B1`: 相同架构但 TSFM backbone 随机初始化，冻结随机 backbone，同 Bridge/Head 参数预算；用于检验预训练表征贡献。
- `B2`: Pretrained TSFM + 标准 LoRA（Q/V 或 Q/K/V/O + FFN 实际支持层选择要注册），相同 Bridge/Head。
- `B3`: 现成轻量电池专用直接轨迹预测模型（历史 M1/M2/A1 的代码在本任务新 split 下重新训练；若暂时不能迁移，就保留旧数值为仅供诊断的历史对照，禁止进入公平 K-shot 主表）。

P0 pilot 先选定1个固定内折、seed42、有限但真实的最大epoch（选模/运行预算写入配置），做微调梯度流和内存测试；正式基线再扩展3 seeds以及预注册目标任务。对所有方法报告：可训练 head、bridge、PEFT、总量；主干参数总量；验证 MAE；同等优化更新量下收敛；峰值 allocated/reserved显存与真实耗时；权重 SHA。不同 LoRA 初始化、target_modules、r、alpha、dropout、weight_decay 必须可追溯。

**P0 Gate：** 只有当官方预训练 checkpoint 被真实加载、Bridge 有明确数据语义、训练/推理闭环与无泄漏测试全部通过，才进入 P1。若预训练没有带来任何增益，不得直接断言不能做：完成 Bridge 与任务一致性诊断，并在报告中注明继续研究需要承受的风险；不为了得到正结论改动隐藏输入。

交付：`P0_REPORT_ZH.md`、`DATA_SPLIT_LOCK.json`、`TSFM_BACKBONE_AUDIT.json`、`BRIDGE_CONTRACT.json`、`P0_METRICS.csv`、`P0_FAILURES.json`。

## 4. P1：退化梯度子空间的可检验性审计（以诊断为主，不立刻训练新模型）

目标：H1 是否成立。仅在**源训练/元验证**数据计算诊断梯度。禁止用最终目标 Query 未来标签选择子空间；目标 Support 中未来标签只能用于最终 K-shot 适配，不能提前混入源域基方向构建。

对每个候选 TSFM 线性矩阵（先 Q/V，后 Q/K/V/O 和 FFN），在相同 checkpoint、相同 trajectory head、相同 loss 下，以每个源工况多颗独立电池计算 `G_l,g = mean_cell ∇_{W_l} L_cell`。暂时开放 `requires_grad` 仅为计算完整矩阵的梯度，不要优化主干；正确恢复冻结状态。保存 grad norm、形状、统计样本数和屏蔽计数。为避免平均梯度方向相反相互抵消，额外分析单电池梯度、组内/组间一致性、梯度能量协方差或低秩随机 SVD，不允许仅使用 mean grad SVD 一个证据。

指标：

- 各层梯度奇异值曲线及能量保留比例，`r∈{2,4,8,16}`（超出层维度时跳过）。
- 工况间主子空间 principal angles、投影重合分数，Bootstrap 稳定性；剔除工况偏差后是否仍稳定。
- 留一源工况的子空间投影能量：`||Uᵀ G_holdout V||_F² / ||G_holdout||_F²`，仅在源 meta-validation 计算。
- 控制组：随机正交 U/V；冻结权重 W0 的前 r 个奇异向量（LoRA-XS 风格）；仅用退化 `trajectory` 梯度；使用随机标签/打乱电池-结局对应关系的负控制（仅源训练可做）；按绝对容量主导与去量纲早期特征分别诊断。
- 与不同层 Grad Norm/投影收益和最终验证误差的关联；不能仅用谱形图替代预测实证。

禁止用测试集合的标签去决定“哪些层该做 PEFT”。用源验证的决策规则选一组统一 target_modules，并冻结配置（例如每层 Q/V 或某些层 FFN），所有方法严格用同一个集合。

**P1 Gate：** H1 的支持标准是：相对随机和权重 SVD 子空间，在源组留出上至少观察到可重复的梯度捕获或下游更新改善；需要报告置信区间/重抽样稳定性，不能从任意一张热图定论。如 H1 不成立，明确写 `H1_NOT_SUPPORTED`，后续允许继续跑 P2 的**对照研究**，但不得声称梯度驱动共享子空间已被证实；P3/P4 大矩阵可暂停。

交付：`P1_GRADIENT_AUDIT_ZH.md`、每层CSV、主角度矩阵、随机对照、SVD 图、配置锁、源 UID 与状态指纹。

## 5. P2：最小新方法——退化子空间低维微调

形式：`W' = W0 + U_l C_l V_l^T`，`U_l∈R^{d_out×r}`, `V_l∈R^{d_in×r}`，目标 trainable `C_l∈R^{r×r}`；U/V 从 P1 的**源训练**退化梯度方向估计，固定；允许在**源训练**上通过 meta-training 进一步学习 U/V，但不得从目标 Query 提取梯度调方向。

注意：`U C Vᵀ` 本身并非新颖；LoRA-XS 已使用冻结 SVD 基方向和小核心矩阵。P2 要比较“退化方向来源的价值”。

同一 TSFM、同一输入 Bridge、同一 Head、同一训练目标、同一目标层，至少比较：

- `LORA`: 标准 LoRA r=2/4/8（alpha/r 与 dropout 按验证预设）。
- `LORA_XS`: 原版或忠实复现的基于 W0 SVD 的冻结 U/V + trainable core；登记 Σ 使用及 scaling。
- `GRAD_LOWDIM`: 基于源训练 gradient SVD 的固定 U/V + trainable core（P2 主要方法）。
- `RANDOM_LOWDIM`: 与上一组相同r、参数、初始化尺度的随机正交 U/V。
- `LORA_GA`: 根据公开论文/代码实现梯度对齐初始化的 LoRA 强对照，真实实现无误后加入；如不可复现标记，不以普通 LoRA 冒充。
- `FULL_FINETUNE`（资源允许时小矩阵/少数层；不同训练参数规模单列资源参考，不作为相同预算对照）。

必须做两种公平对照：A 相同 rank；B 相近**实际可训练参数**（若完全相等做不到，记录差异并按参数量作误差曲线）；同时报告包括 Head/Bridge 的总可训练参数，避免展示“16 vs 4096”却遗漏其他组件。统一优化更新总次数、同一验证选权/超参数搜索预算；每种方法可在**相同允许的验证搜索预算**内调 lr，不能强行所有方法使用对其显著不利的单一 lr。

实现工程自检：`W0` 冻结、U/V 冻结、C 有有限非零梯度；C=0 产生与冻结模型同等前向（如存在源阶段 base C0 要相应定义）；数值高精度对照；可恢复 checkpoint；多层不同形状；完整 query 和 mask；CPU/CUDA。

**P2 Gate：** 若 GRAD_LOWDIM 不能在源开发与独立源组留出上超过 LoRA-XS 或随机子空间（考虑统计不确定性），保留负结果，不能以复杂 P3 的收益追认 P2 的子空间创新。也可转为“纯目标域受约束适配”的假设，但需注册新版本。若有价值，继续 P3。

交付：`P2_RESULTS_ZH.md`、`P2_METRICS.csv`、`P2_PARAM_BUDGET.csv`、模型检查、逐电池预测和误差反例。

## 6. P3：物理双视角条件化低秩更新

在 P2 冻结子空间版本基础上设计（只在验证范围选择）：

```
W'_l(x) = W0_l + U_l · D_drift(x) · C_l · D_abs(x) · V_l^T
```

其中 `D_drift`/`D_abs` 为 r×r 对角门控，始终只利用**该颗电池前100圈 VIQ + 可用的已知 policy**。绝对响应分支应包括合法的电压/电流/电荷水平及预先知道的策略；相对变化分支使用前缀内 `Q_n(V)-Q_ref(V)`、全局/局部残差和稳健跨循环斜率、mask/质量分数。不能将载荷响应分解结果直接解释为电极级 LLI/LAM/SEI 真值。

必须确保同一电池不同样本的 dynamic ΔW 正确处理 batch，不能把 batch gate 先平均后共用；检查梯度是否到达各分支。U/V 及主干 W0 在目标域保持冻结；具体哪些 gating network 是源训练还是目标训练在合同中注明。

消融（参数预算匹配或附等预算替代 MLP）：

- `P2` 无门控；
- `ABS_ONLY` 仅绝对响应；
- `DRIFT_ONLY` 仅相对变化；
- `TWO_VIEW` 两者；
- `SHUFFLED_DRIFT` 将不同电池之间 drift 特征打乱，仅作诊断，不能当真实训练增强；
- `STATIC_GATE` 使用同等参数但与输入无关的固定可训练 gating；
- `NO_LOCAL` 去掉显式局部响应分解，但保留原始 VIQ，明确消融范围。

检查不同化学体系下 VIQ 电压范围与尺度差异；若不同测量条件下电压 bin 无法对应，须各域单独合法编码再投影到共同嵌入，不能用相同 bin 假装物理等价。

**P3 Gate：** 在源开发任务上，TWO_VIEW 是否优于 P2/STATIC_GATE；随机/打乱控制是否削弱性能；改善是否主要出现在有真实局部响应支持的数据。若缺乏增量，不合并动态门控进最终方法，保留 P2 结构进入 P4。

交付：`P3_CONDITIONING_ZH.md`、单/双门控完整消融及显式质量掩码对照、作用于真实层的适配矩阵及梯度验证。

## 7. P4：真正的 Few-Shot 支持集校准（核心）

在 source-pretrained 结构化适配器上，以目标域 K 颗完整轨迹作为 Support，**只调整目标域 low-dim core `δC_d`**，禁止在 K-shot 中暗中更新主干、Bridge、Head、U/V、源门控；若需额外策略，定义单独 baseline。支持池中的每颗电池只能以其真实已观察 future-SOH 作为监督，不能把无观测后缀补全为标签。目标 Query 永远不参与微调。

候选校准：

1. `GRAD_CORE`: trainable δC，用 AdamW、L2 收缩并以 source core 初始化。训练步数/学习率/验证选择必须仅依赖源 meta-validation 或 Support 内合法留一评估。
2. `JACOBIAN_RIDGE`: 在源 core 点对支持集预测线性化 `y(δc) ≈ y0 + Jδc`，解 `(JᵀΩJ+λI)δc=JᵀΩ(y-y0)`；Ω 保证**电芯等权**且考虑时序点高相关。做条件数/奇异值监控，使用 `torch.linalg.solve`/稳定 SVD，不直接求逆。该解仅是局部线性近似，不能伪称全模型闭式最优。
3. `SHRINKAGE`: `C_d=C0+γ_d δC_d`。γ 由源 meta-validation 的支持集拟合诊断/前缀可比性确定，不得使用目标 Query 未来标签调 γ。
4. `META_EPISODIC`：只有在 source 域可构造足够独立 Episode 时，模拟 Support→Query 学习 U/V、source core 初始化或校准超参数。Meta-train/meta-val/test 均 UID/协议互斥；若无法构造足量真实 Episode，跳过且诚实说明。

K-shot强基线：Frozen + Head；source LoRA 0-shot；source LoRA + 目标K-shot继续 LoRA；LoRA-XS + 目标K-shot core；P2 的 gradient lowdim core；P3 full two-view core；简单物理特征 + Ridge/GPR 小样本校准；适当允许 LoRA-GA。同一查询池、同一支持 UID/完整轨迹输入预算。要区分`source training`与`target adaptation`两个阶段的可训练参数；即使目标 LoRA 只调<1%总参数，也必须报告真实值。

每个 K 至少记录5份固定 support draw（若样本不足则实际可行 draw 数），建议3个训练种子（42、123、2026）；大矩阵可先在一主干、2个有支持K值上做 pilot，然后预先冻结后展开全部 K。重复抽样共享同一个固定 Query 集，使 K-shot 曲线可比；不可为了选出最好的 draw 只报告最优值。

主效应：相对标准 LoRA 与 LoRA-XS 的 Query MAE 差、按物理电芯配对的置信区间、按协议组分层的敏感性、negative-transfer rate（相对 0-shot 源预测变差的电芯比例）、支持集数量曲线及适配自由度。额外干预：支持电池标签在 source 合法范围内打乱；支持集加入与 Query 不同工况的电芯；将 drift 特征打乱；只比较预测前可信信息控制。

**P4 Gate：** 必须证明 K>0 的收益确实来自目标 Support，而不是 source 训练规模、预测头改动或额外 target 数据。1-shot/3-shot 下若优于 LoRA 但不优于 LoRA-XS，表述应是“低维参数数目有益”，不能把它归功于退化结构。负结果保留原值。

交付：`P4_FEWSHOT_ZH.md`、`P4_SHOT_CURVES.csv`、`P4_NEGATIVE_TRANSFER.csv`、全部 episode IDs、seed、checkpoint、target-only updates、支持集目标容量与有效 masks 的摘要、测试逐电池预测。

## 8. P5：论文级外部评测、完整消融与理论边界

P5 必须在 P0–P4 的架构和超参数**完全固定后**运行正式独立评测。支持集由预先锁定随机种子确定，Query 集不参与选择。下游预测指标：完整轨迹 MAE/RMSE（每电芯等权）；四时距分箱（每箱支持数）；名义 EOL80 仅在有实际事件/删失证据且定义一致时评分；运行时与有效可训练参数；显存与参数大小；跨工况/OOD 分层误差。

统计：对物理电芯配对 bootstrap（至少2000次）与条件允许时的工况组 cluster bootstrap；对同一组 source-seed/support-draw 平均与方差作合适分层，区分 fixed-prediction bootstrap 与 train-seed 方差；不将一个 seed 的样本 bootstrap 伪称跨训练随机性的保证。所有有效分母、缺失和 right-censor 均入表。对多重比较优先标明 exploratory，避免逐项挑显著结论。

重点复核 MATR 历史失败电芯 `MATR_b1c2` 和任何新观察到的高误差/下界截断实例；保存 source 0-shot、LoRA、LoRA-XS、结构化 PEFT、支持集校准的轨迹叠加与速率变化。不能仅输出均值提升，不报告失效个体。

论文创新审计必须对照：LoRA、LoRA-GA、LoRA-XS、HiRA、ABBA、Time-PEFT、公开的 Timer+LoRA 电池容量退化工作、BatteryMFormer、PBT；列出“已有人做过”“本方法不同”“实验能否证明差异”。如果发现所谓公式已经现成存在，要调整 novelty 论述，不得改名后冒充新数学方法。

理论分析只允许写明确有条件的数学命题，例如正交投影最优低秩近似、局部 Jacobian-Ridge 正则估计的条件数/收缩意义；不允许由轨迹低维直接推断权重梯度共享低维，也不得对非线性网络给出未经证明的泛化界。

交付：`P5_FINAL_REPORT_ZH.md`、完整 markdown/CSV 主表、所有消融与分域主表、主图（不叠加图例/文字）、核心代码与依赖版本、创新边界和 LIMITATIONS。

## 9. 统一工程验收与复现合同

每个正式 run 输出：

```
RUN_CONFIG.json
SOURCE_SNAPSHOT.json
DATA_FINGERPRINT.json
EPISODE_IDS.json
BACKBONE_IDENTITY.json
TRAINABLE_PARAMETER_REPORT.json
TRAIN_LOG.jsonl
CHECKPOINT_SELECTED.pt
CHECKPOINT_SELECTION.json
TEST_PREDICTIONS.parquet
TEST_CELL_METRICS.csv
RESULT_METRICS.json
COMPLETED.json
```

`COMPLETED.json` 只有经过正常训练完成、选择权重冻结、测试入口真正执行、独立结果重放和统计重算后才能生成；仅 smoke 或测试失败不能标 COMPLETED。

必须提供以下单元/集成/反例检查：

1. 物理电芯/续测 UID 互斥及被剔除或不足支持的分母。
2. 目标 Query 未来 SOH 扰动/删除不会改变前缀输入、标准化、动态门控或适配权重；只有在显式计算评测指标时读取真实 Query 结局。
3. Support 标签变化可以影响 target core，但不能影响 source stage 的冻结 U/V、W0 或原始 data freeze。
4. 所有方法的预测头、输入 Bridge、Query 网格、归一化范围、目标 mask 和选权规则逐项相等。
5. LoRA 与拟议方法在低秩张量单层 Oracle 上具有正确计算图、非零梯度、模型重载一致性。
6. 批内不同电池的动态门控不会串用；可在保留掩码下对调输入，确认每颗输出按预期变化。
7. `random_init` 与 `pretrained` 同架构真实参数来源可查；预训练 checkpoint 的 tensor key/checksum 能认证。
8. 重跑同一 run：同环境同随机状态进行位级/容差复验，GPU 不确定算子按合同记录；不能把训练日志一致性替代完整预测重放。
9. 预测可不以实际观测的 T_i 为条件生成固定查询网格；mask 后计算指标；不能借助 Query 真值控制序列长度。
10. 总参数数、trainable 参数、optimizer 参数状态和可恢复 checkpoint 实际对应；断点恢复不得重复优化步骤。

每阶段运行入口提供 `--config`, `--data-root`, `--output-root`, `--device`, `--seed`, `--resume`, `--dry-run` 等明确参数（只在有语义时启用），`--help` 可运行。训练要有显存安全策略（单卡4090/4090D优先；可设置梯度累积、activation checkpointing；严格标注 batch/累积/AMP/TF32），先做单卡 smoke 再启动正式。若遇到 CUDA 内存不足，先报告真实显存峰值和降载依据；不得用降低模型规模而不记录的方法影响与 Baseline 的公平性。

## 10. 推荐计算调度与预注册停止规则

不预先承诺具体耗时或显存需求；先通过 P0 的实测计算预算核算。建议按如下顺序消耗算力：

- **小预算 P0/P1**：完成数据与模型完整性、一个主干的1折真实 pilot、关键层梯度子空间；每阶段提交阶段报告。
- **中预算 P2**：Moirai small 上的 LoRA / LoRA-XS / Gradient-core / Random-core，r=2/4/8，小规模 source meta-validation，至少3 seeds后判断。
- **中预算 P3**：仅对 P2 最可信的 rank 和 target_modules 进行 ABS/DRIFT/TWO_VIEW 门控消融；不要将所有 rank×层×k 全笛卡尔积暴力跑完。
- **大预算 P4/P5**：确认正向证据之后再运行5个 K-shot 值、多种支持集抽样和外域测试；第二 TSFM 主干留到方法确定后。

严禁以单个测试结果作为继续/停止依据。依预注册的**源验证**证据决定进入下一阶段；对失败机制可做额外诊断，但需登记为新实验版本。即使新模型不优于 LoRA-XS，仍必须报告原始负结果和可复现失败原因，不得更换评测任务直到优势出现。

## 11. 最终交付格式（必须形成真实文件）

报告至少包含以下事实表：

- `DATA_READINESS`：各域总电芯数、K100可用数、完整未来 SOH 有效数、原始/供应循环认证、标签及 EOL 支持、每种K-shot可用 Episode 数、目标域不可用原因。
- `TSFM_BASELINE`：真实 backbone 型号、预训练 vs 随机初始化、Bridge 和 Head、参数数、Source-val与目标0-shot误差。
- `GRADIENT_SUBSPACE`：各层有效秩、主角度、source-group 留出捕获、random/W-SVD 对照。
- `PEFT_COMPARISON`：同rank和等可训练参数两套表，统一 Bridge/Head 的多seed、分工况 MAE/RMSE。
- `FEWSHOT_SCALING`：K=0/1/3/5/10，在固定 Query 下不同 Support draw 的分层结果及负迁移率。
- `ABLATION`：NO-GATE、ABS、DRIFT、TWO_VIEW、Random/Shuffle、No source gradient、No shrinkage、No support。
- `COMPUTE`：峰值显存、实测GPU时间、优化更新数、训练和推理时间、源/目标微调的可训练参数。
- `FAILURES`：至少列出所有性能恶化电芯及严重外推失败案例。

最终提交一个 `report/drs_peft/MASTER_README_ZH.md`，指向各阶段报告和命令。按以下结构在聊天给我报告，不要只发“已完成”：

1. 实际完成阶段与状态（PASSED / NEGATIVE / BLOCKED / PARTIAL），最重要的三个数值证据；
2. 新增/修改文件清单及相应 git commit；
3. 每阶段运行的真实命令、GPU 型号、可训练参数、实际数据分母；
4. 是否证明 H1/H2/H3，每一项是证据支持、证据不足还是被反证；
5. 关键 Baseline 相比误差和置信区间（无结果不得填数）；
6. 主要失败、物理解释边界和下一个**由当前结果驱动**的实验步骤。

## 12. 立即开始执行的步骤（请在当前 Codex 工作区真的做）

**先执行：**

A. 检查仓库状态、上述16份材料和现有数据。生成 `REPOSITORY_READINESS_ZH.md`。

B. 构建并核验 Few-Shot Episode 构造程序与冻结配置；列出 MATR 未见协议、Stanford 目标数据集和 HUST 目标数据集上每个 K 的**实际可用**支持/查询数量；若任何目标需要恢复原始资料，只标记其资格，不虚构。

C. 审计官方预训练 Moirai checkpoint/forward 接口，做两条 Input Bridge 的真实前向和泄漏 QA，锁定一条满足任务的 Bridge。

D. 运行 P0 初始真实 Baseline，包括预训练冻结、随机冻结、LoRA；保存可重复结果。

E. 完成 P1 源数据退化梯度子空间诊断（含随机与 W-SVD 控制），给出具体 H1 判断。只有通过预设 Gate 才启动后续 P2–P5；没通过也交付负结果。

F. 若资源与阶段 Gate 均满足，按照上述顺序运行剩余阶段。**不要先声称完成，再补空文件。**

最后，请对所有结果进行严格审计：同一任务、相同输入预算、相同预测头、相同支持集、相同选权规则、没有 Query 未来泄漏、没有测试调参、可重复重放。任何不成立的比较必须从主结论中删除并记录原因。
