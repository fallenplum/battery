# DRS-PEFT 方法与先行研究审计

本审计依据2026-11-17用户提供的DRS-PEFT合同，逐项读取P0–P5及统一验收约束。本文描述源代码、数学定义和预注册门槛，实验结论由后续真实训练产物决定。尚未据此宣称H1/H2/H3成立。

## 研究对象与计算图

主任务为单颗Query电池前100个认证原始或明确声明的供应循环VIQ、可信policy，配合K颗独立Support电池的全部真实可观察未来轨迹，预测固定网格未来SOH。主误差按每颗Query的真实观测点先平均，再跨物理电池等权。

查询网格由域级资格审计预先固定。合同的101–2500属于示例，HUST已有观测延伸到2689圈。完整未来主指标需要固定上限覆盖所有已资格认证的观测，单颗Query的未来长度只进入评分mask。

P0–P4所有PEFT比较共享选定Bridge、连续cycle-query Head、SOH100锚点、数据角色、loss、验证选权和固定Query集合。B0/B1各自在同Bridge内部匹配可训练预算，分别使用真实预训练与真实随机初始化主干。原生多变量与cycle-embedding桥接各需真实前向、CPU/CUDA、batch1/8、缺失mask及未来扰动检查。

## 先行方法的实际定义

### 标准LoRA

对PyTorch权重W0∈R^(m×n)，增量为sBA，其中A∈R^(r×n)、B∈R^(m×r)、s=alpha/r。A使用Kaiming初始化，B初始化为0。仅A/B可训练，主干冻结。单矩阵适配参数为r(m+n)。本地既有LoRA代码位于项目外层src/models/moirai_battery_adapter/lora.py，本轮独立命名空间保持相同基本算子并增加源/目标作用域核验。[LoRA论文](https://arxiv.org/abs/2106.09685)

### LoRA-XS

作者代码对W0.T进行TruncatedSVD。在PyTorch权重方向下，相当于B=U、A=Sigma V.T，实际更新为s U R Sigma V.T。A/B冻结，可训练R为r×r，作者代码采用std=1e-5的Gaussian初始化。Sigma影响核心优化的度量与尺度。单纯正交U C V.T配零核心应明确标为weight-SVD core。双方函数空间可能相同，优化条件、收缩几何与初值需要逐项登记。[官方初始化](https://github.com/MohammadrezaBanaei/LoRA-XS/blob/main/utils/initialization_utils.py)、[官方SVD](https://github.com/MohammadrezaBanaei/LoRA-XS/blob/main/utils/svd_utils.py)、[官方前向](https://github.com/MohammadrezaBanaei/LoRA-XS/blob/main/utils/latent_utils.py)

本轮LoRAXSLinear显式保留Sigma与Gaussian核心，并使用CPU float64完整SVD替代作者的随机TruncatedSVD。该数值实现差异写入方法身份，忠实的更新算子、参数规模和缩放另行验证。

### LoRA-GA

作者方法从真实完整权重梯度SVD初始化两个非零低秩因子，ArB2r使用右奇异方向前r个与左奇异方向第r至2r个，stable缩放按out_features^(1/4)/sqrt(gamma)作用于两因子。初始非零更新必须通过W_init=W_pre−sB_init A_init补偿，最终相对预训练的更新为s(B_final A_final−B_init A_init)。只做梯度SVD初始化而省略补偿会改变模型起点。[官方说明](https://github.com/Outsider565/LoRA-GA/blob/main/doc/detail.md)、[官方算子](https://github.com/Outsider565/LoRA-GA/blob/main/peft/src/peft/tuners/lora/layer.py)

本轮LoRAGALinear冻结原W_pre，存储初始A/B两个小型buffer，通过当前低秩响应减初始低秩响应实现同一补偿。实现使用精确float64 SVD，并支持ArB2r、A2rBr、ArBr及stable/unit/gd缩放。dropout=0或eval模式下，初始前向与原模型一致。保存当前因子和初始因子，保持完整补偿重放。

### 额外直接相关的LoRA-SB

LoRA-SB已使用由full-FT梯度或首步更新导出的冻结正交基，并只训练r×r中间核心。它比仅W0-SVD的LoRA-XS更接近P2的gradient-core。P2的裸参数化与梯度SVD来源均已有先例。新的研究价值需要来自跨源工况的稳定退化方向、有效的电池响应条件化与严格目标少样本校准，并通过对应对照证明。[论文](https://arxiv.org/abs/2411.19557)、[官方实现](https://github.com/CERT-Lab/lora-sb)

已使用可用网页检索与作者原始代码交叉核对。直接Google搜索页在当前工具中不可访问，未以搜索摘要替代官方算子审计。

## 模块API与身份

实现文件为models/drs_peft/peft.py。

```python
install_peft(model, method, rank, alpha=None, dropout=0.0,
             target_names=("q_proj", "v_proj"), bases=None,
             gradients=None, seed=42,
             ga_direction="ArB2r", ga_scale="stable", ga_gamma=16.0)
random_orthogonal_bases(out_dim, in_dim, rank, seed=42,
                        dtype=torch.float32, device="cpu")
svd_bases(matrix, rank)
peft_context(model, abs_gate=None, drift_gate=None)
begin_target_adaptation(model, mode="core")
adaptation_parameters(model)
trainable_parameter_report(model)
```

method包含lora、random_core、weight_svd_core、grad_core、lora_xs、lora_ga。grad_core需要按真实完整模块名提供源训练U/V，lora_ga需要真实源梯度。模块注入前构造所有替换，校验失败恢复原requires_grad状态。局部初始化仅使用隔离的CPU生成器，避免修改全局CPU/CUDA训练随机态。

随机、W-SVD与grad-core的U/V为冻结buffer，core是唯一适配Parameter。dynamic路径为U D_drift C D_abs V.T，门控必须为每颗物理电池独立的[B,r]，支持[B,T,d]与[B,d]。ContextVar隔离嵌套和并发上下文，正向与反向均能保留门控梯度。开启activation checkpointing时，门控context应覆盖正向与反向的重计算。

目标core适配采用C_eff=C_source+delta_core，delta_core初始严格为0。传入整个模型调用begin_target_adaptation后，Bridge、Head、主干、source core和门控参数全部冻结，仅delta_core可训练。重复调用保留当前delta，checkpoint加载可自动恢复该作用域。LoRA继续训练由显式mode=lora单独启用A/B。数值检查不能把标准LoRA初始A梯度为0或C=0时门控梯度为0误判为错误，训练后的C0或注册warmup用于P3门控梯度核验。

## 实际参数预算

QV矩阵的实际审计维度为384×384，六层共12个目标矩阵。以下均为单矩阵有效可训练数量。

| 比较 | LoRA rank | LoRA参数 | core rank | core参数 | core相对差异 |
| --- | --- | --- | --- | --- | --- |
| 同rank2 | 2 | 1536 | 2 | 4 | 按不同预算报告 |
| 同rank4 | 4 | 3072 | 4 | 16 | 按不同预算报告 |
| 同rank8 | 8 | 6144 | 8 | 64 | 按不同预算报告 |
| 近参数1 | 2 | 1536 | 40 | 1600 | +4.17% |
| 近参数2 | 4 | 3072 | 56 | 3136 | +2.08% |
| 近参数3 | 8 | 6144 | 80 | 6400 | +4.17% |

Head和Bridge的训练参数应加到源阶段总预算中。P4冻结这些组件后，应重新给出实际目标适配参数。冻结U/V、LoRA-GA初始补偿因子和主干仍有存储与计算开销，需要单列。预算曲线使用有限注册子集，避免rank×层×K的完整笛卡尔积。方法可有不同LR，但验证试验次数、优化更新量、选权规则和标签可见性一致。

## 分阶段门槛与停止

数值门槛见configs/drs_peft/stage_gates_v1.json，须由主执行者在读取本轮诊断结果前冻结。

| 阶段 | 核心证据 | 放行与失败处理 |
| --- | --- | --- |
| P0 | 真实官方权重、两桥语义、真实baseline训练、未来扰动QA与重放 | 完整闭环后进入P1，预训练无收益时完成接口诊断并记录风险 |
| P1 | 源组留出的稳定投影捕获或真实下游更新收益，random/W-SVD及打乱控制 | 单seed只标PARTIAL，正式三seed门槛见JSON，失败可继续P2受控研究，P3/P4大矩阵停止 |
| P2 | 同rank及近参数的LoRA/XS/W-SVD/gradient/random/GA | 三seed受控收益达到预设幅度与配对区间要求后进入P3，阴性结果保留 |
| P3 | ABS、DRIFT、TWO_VIEW、STATIC、SHUFFLE、NO_LOCAL | TWO_VIEW需在参数控制下优于P2/STATIC并具干预响应，阴性时P4保留P2 |
| P4 | 固定Query、多Support draw、仅delta_core的真实完整轨迹校准 | 同时比较自身0-shot、LoRA与XS，单独胜LoRA只支持低维适配解释 |
| P5 | 全结构与超参冻结后的外域评测、分层不确定性与失败个体 | 仍受历史暴露、供应循环语义、删失与多重比较限制 |

当前量化选择为P1主rank4，投影捕获相对random和W-SVD的平均绝对差至少0.02且组bootstrap下界大于0，bootstrap投影重合均值至少0.5，三source seed方向一致。可替代证据为源留出真实更新的平均相对MAE改善至少1%且配对cell/组区间下界大于0。P2/P3的核心相对MAE门槛同为1%，所有配置在结果查看前冻结。阈值属于本轮执行决定，不属于已证明理论定理。

## 数学与泄漏陷阱

- 投影能量公式要求正交U/V，含Sigma的LoRA-XS因子应先取正交基再计算。否则所谓能量比例可能超过1。
- 单电池线性层梯度的秩受token数量约束，低秩谱本身无法证明电池退化共享结构。平均梯度还可能抵消，需逐cell、组内/组间和源组留出控制。
- 留出梯度可用于诊断投影捕获，下游更新方向应由源训练构造。先用留出标签更新，再在同一组上评分会夸大收益。
- 两个diagonal gate与核心存在尺度重参数化自由度。消融可检验条件化价值，单个gate数值无法唯一解释真实电化学分量。
- P4的delta初值为0。把source core同时当delta初值再相加会翻倍，K=0须保持源模型预测。
- Jacobian-Ridge是局部加权平方拟合，主指标仍为cell-mean MAE。每cell的观测权重需归一，相关时间点处理由source meta-validation冻结。1-shot的超参依赖源meta验证，不能对唯一Support的未来点随机切分后当成独立电池留一。
- Query未来长度、EOL、目标mask或target标签只能在评分入口读取。Support未来标签仅用于目标适配，不能回流构造source U/V。
- support draw共享Query，重复draw具有相关性。分层bootstrap和seed/draw波动须分开，循环点不属于独立样本。HUST精确协议单例缺少协议内重复。
- 主干临时requires_grad用于P1时，要恢复冻结标志并防止AdamW意外更新。优化器参数ID与允许训练名逐一一致，完整重放由主执行入口验证。

## 工程检查与当前边界

tests/drs_peft/test_peft.py在本机tslib环境完整运行21项测试通过，覆盖CPU/CUDA、batch1/8、显式矩阵oracle、零增量等价、核心/门控梯度、目标delta唯一作用域、state_dict恢复、上下文恢复和批置换、XS的Sigma、GA补偿、参数匹配与安装失败回滚。P1真梯度toy覆盖source-only目标哨兵、极端目标未来扰动、零梯度无定义项、缓存hash、首尾重算及临时flags恢复。三seed门槛、label-shuffle证据边界、P0阴性工程放行和P2两预算负支也有合成fixture验证。P2执行fixture完整尝试18个注册LR trial，覆盖nonfinite、CUDA OOM、必要方法全部LR失败、formal seed子集、共享注册只读与重复执行恢复。真实AdamW微测核对了step观察计数、optimizer state.step和函数恢复。上述测试用于算子与执行边界的工程验收。

run_p1.py复用正式P0 B0的官方权重与源头部，主QV每seed315个逐cell采集、10个首尾重算，另有qkvo_ffn source12/meta8流式次要描述性诊断，总计345个单cell前后向、0优化器更新。次要诊断只保留源均值、谱、留出投影、IDs与hash，缺少完整candidate协方差和重采样推断，禁止据此更改主QV目标或rank。LoRA-GA使用的mean_gradients为physical-cell等权，P1 mean-SVD的group_mean_gradients单独按exact protocol等权保存。

P2预注册矩阵在H1阴性时仍包含同rank4六方法与近实际参数core56四方法的三seed受控比较，额外rank2/8停止。每方法三LR在seed42上各60epoch搜索后冻结LR，所有正式seed各120epoch。明确定义的nonfinite/CUDA OOM登记失败trial与真实已完成step，其他身份或hash错误直接中止。必要方法三个LR全部失败时标BLOCKED。最终目标Query标签参与训练、梯度、选模或评分计算为0，整cohort数据文件可以在内存加载。

真实P1三source seed合计1035次单cell前后向、0优化器更新，主QV共享低维捕获门槛通过。label-shuffle控制的捕获未弱化，结局特异方向和电池退化特异机制仍缺少对应支持。P2的18个搜索及66个正式单元全部闭环，共99000次源训练更新。梯度核心相对LoRA-XS的r4及近参数core56平均相对MAE改善分别为0.5812%与0.8510%，配对cell及protocol区间均跨0，三个seed的方向存在分歧。P2门槛结果为NEGATIVE，后续P3/P4扩展放行标志为False。H2、H3保持未评估状态。

首个LoRA r4 trial的660次更新、epoch55选权、冻结参数身份及磁盘checkpoint已独立复核。CPU重建并重放30颗source-meta-validation电池，完整固定网格预测的CPU/GPU最大差为1.40667e-5，符合既有CPU重放容差1e-4。此复核只使用源训练和元验证允许的信息。

优势为参数作用域、原始W0身份和条件化更新有明确可核对的实现，强对照算子已有原始代码参照。风险包括Σ条件数、source group有限支持、门控尺度自由度、label-shuffle未弱化及固定子空间的外域失配。实现算子与已完成单元身份核验的置信度为高，H2、H3及少样本新颖性结论仍需完整受控比较与冻结后的目标域结果。
