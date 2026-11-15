# M4独立消融执行合同

日期为2026-11-16。本合同固化本轮用户授权的A1至A4独立消融，以及已完成M4所使用的共同训练与评估原则。四项均以完整M4为直接基准，各自移除或替换指定机制，按A1全部六单元、A2全部六单元、A3全部六单元、A4全部六单元顺序执行。

## 数据、任务与冻结范围

共同研究问题为MATR169颗LFP/石墨电芯的真实循环1至100、共同4C放电窗2.8至3.3V的VIQ及已知快充policy，预测全部实际可观测未来SOH。保留原名义容量1.1Ah的SOH口径、100圈前缀锚点、64点电压网格、缺失mask、20维global、64点local residual、8段26维local features、15维policy与所有冻结标签。

直接复用`data/unified_matr_seed42/`及原`load_frozen_data`守卫、原五折cell-disjoint split和原unseen-protocol sensitivity。全部输入、标签、数据冻结与split指纹与M4记录核对。旧七臂源码、M4八个注册依赖、冻结数据和历史结果保持原指纹。所有新实现与结果写入新增文件及`results/unified_matr_ablations_seed42/`。

每项消融均从seed42随机初始化训练。共享结构与张量使用M4相同的隔离seed初始化，新增或替换结构的初始化明确登记。每项的全部参数、可训练参数与实际计算图路径按arm记录，保留结构性移除所产生的参数量差异。

## 四项独立机制替换

| 消融 | 唯一指定变化 | 共同保留的主要路径 |
|---|---|---|
| A1_M4_LEGACY_FLOW | 用原`LegacyFlowHead`恢复原始解析Flow，替换非负损失速率积分头 | I1、动态rho、持续状态、质量、两层真实B调制、raw/reference VIQ、policy、state/condition与观测层 |
| A2_M4_NO_B_WRITE | 两层history均令`Bprime=B`，关闭响应条件化B写入 | 动态rho、持续状态、质量、局部物理、content/selector conditioning、局部self-reference Mamba与原非负rate头 |
| A3_M4_STATIC_RHO | 用64通道静态可学习logits替换rho网络，各通道初始sigmoid值为0.9 | 当前响应、首有效圈初始化、实际gap指数、EMA/mask、质量、gate/B调制、原decoder与所有其他输入 |
| A4_M4_NO_LOCAL_PHYSICS | 禁用显式local residual与local features及其mask/质量噪声通路 | raw/reference VIQ、point mask、global20及global mask、原局部self-reference Mamba、动态持续性、B调制与原rate头 |

### A1原始Flow

`LegacyFlowHead`按旧实现接收16维state、16维condition与`tau=(t-100)/1000`，解析状态演化生成相对100圈的SOH修正。保留`trend(100)=anchor`、观测残差`0.03*tanh(epsilon)`及最终SOH范围`[0,1.2]`。共享编码器、持续性和B写入路径保持与M4同构同初始化。

原始Flow允许回升。逐电芯记录原始trend及最终预测的上升次数与幅度，保留原始预测。两段rate辅助监督始终由预测SOH端点差除以实际cycle间隔计算，窗口为200至500及500至1000。A1的signed endpoint rate可取负值。为兼容共同输出接口，signed loss定义为`-correction`，signed rate诊断定义为`-dcorrection/dtau`，其per-cycle换算继续除以1000。该诊断来自原Flow导数，实际预测严格保留原解析Flow输出，验收允许signed rate及非单调trend。

### A2关闭B写入

仅移除两层history中的`gate*tanh(W_B*z)`写入项。持久状态仍进入既有selector条件化路径，原动态rho、I1物理编码器、质量构造与持久性penalty继续存在。以实际卷积后B和传入SSD的B验证`Bprime=B`。共享张量与M4初始化一致，移除或冻结的gate/W_B参数单列登记。

### A3静态可学习持续性

`rho=sigmoid(logits64)`，初始logits为`logit(0.9)`，保留原rho范围`[0.01,0.99]`。rho本身按通道学习，训练中对所有电芯与循环共用。当前响应、policy、质量、cycle编号及gap均不进入rho生成。

状态转换仍使用`rho**actual_gap`，有效圈的实际gap、首个有效圈令`z=phi`、缺失圈持有状态、invalid输入零梯度及原持久性penalty定义继续使用。质量仍用于B gate，持续状态仍用于B写入和selector conditioning。

### A4移除显式局部物理支路

local residual、local features及其mask在前向图中不贡献物理token、rho上下文或gate质量。局部支持比例、noise代理与其mask对应的质量坐标禁用。global20及其mask继续通过既有全局物理编码路径。

删除physics中的residual/local/bin/local-projection模块。global_encoder逐值使用M4初始化，combine首层由128输入裁为64个global输入，复制M4权重前64列及原bias，保留后续GELU与LayerNorm，数值等价于原combine的局部物理token贡献为零。rho输入从172裁为140，移除32个quality坐标，保留其余列的原M4权重。两层B gate输入从96裁为64，保留persistent-state64的原权重与bias。其他隐藏维度、状态维度与decoder保持原结构及初始化。参数量变化来自移除信息及其连接，未引入容量匹配网络。

raw/reference VIQ与point mask继续进入原frontend及局部self-reference Mamba，必须通过受控扰动和梯度验证其实际作用。A4移除信息范围限定为显式局部物理特征支路，原始VIQ仍包含局部曲线信息。旧`M3-NL`对应的其他结构变化与本轮A4要求分别记录。

## 共同训练与选择规则

- seed42，每单元150epochs，batch8，AdamW学习率3e-4，weight decay0.01，betas为0.9/0.999，eps为1e-8，梯度范数裁剪1。
- scaler仅拟合当前split的train电芯，使用原loader、训练ID顺序、样本排列与标签mask。
- trajectory loss为每电芯全部有效未来观测的MAE，再对有支持电芯平均。rate auxiliary按每电芯可用窗口平均，乘1000后权重0.1。持久性penalty保持原实际gap归一化平方变化定义与系数0.001。
- 按validation cell-mean full trajectory MAE的strict first-min选择checkpoint，完成150epochs，随后冻结checkpoint并评分对应test。
- A1与其他arm共同使用prediction endpoints计算rate auxiliary。全部arm保持相同trajectory、rate、EOL与协议统计接口。
- 复用M4专属的单设备autograd scan修复，保持跨设备scan数学值及其伴随梯度一致。旧注册源码保持原SHA。

## 矩阵与更新总账

| 范围 | 单元数 | 每单元epochs | 计划optimizer updates |
|---|---|---|---|
| A1五fold及unseen-protocol | 6 | 150 | 12450 |
| A2五fold及unseen-protocol | 6 | 150 | 12450 |
| A3五fold及unseen-protocol | 6 | 150 | 12450 |
| A4五fold及unseen-protocol | 6 | 150 | 12450 |
| 四项主矩阵合计 | 24 | 150 | 49800 |
| 每arm fold0完整训练复现 | 4 | 150 | 8400 |
| 每arm fold0工程smoke | 4 | 1 | 56 |
| 实际优化预算合计 | 32 | 分项列示 | 58256 |

每个arm五fold各2100updates，unseen-protocol为1950updates。先完成四个arm的全部24个主矩阵单元，再按A1至A4执行各自fold0完整重复训练。每次完整复现2100updates，每次工程smoke14updates。工程控制的前反向不执行optimizer更新。重复训练用于核对完整训练复现，预测比较按原24个主单元统计，每个电芯的主结果分母保持原定义。

## 独立实现验证

新增验证入口为`scripts/verify_m4_ablation_contract.py`。工程测试使用合成控制及真实冻结train-prefix，不执行正式模型选择。CPU先核计算图，CUDA由主执行者协调串行。

- 按共享state_dict键核同初始化与参数差异，逐arm记录唯一被替换组件、保留组件、requires_grad参数与工程反向可达参数。
- A1以旧Flow同权重正常入口逐值对照，核anchor、query集合与顺序一致性，记录允许回升的结构事实与真实观察。
- A2捕获两层history实际post-conv B与SSD接收B，验证关闭写入后恒等。对M4的W_B采用固定受控扰动，验证A2的非B写入条件化路径继续发挥作用。
- A3验证rho在响应、quality、policy与cycle坐标扰动下保持同一通道值，logits可反向学习，actual gap仍作用于EMA。
- A4在被移除的local值和mask上施加极值、NaN或翻转，验证正常预测不变及输入梯度为0。另对raw/reference VIQ、global与policy进行合法受控扰动，验证被保留输入真实作用。
- 每arm验证first-valid、invalid hold/零梯度、prefix时序因果性、输入cycle1至100界限、真实train前反向有限与fresh seed42首批复现。
- 独立标量EMA与SSD oracle、原M4数据/来源指纹核查及源码执行前后SHA一致性保留机器收据。

## 结果、优势与风险

结果保存每arm的合同、smoke、六套训练日志、optimizer更新数、validation first-min选择凭证、checkpoint、逐电芯trajectory/rate/EOL预测、协议支持与误差、异常记录及正常入口重放。共同主指标为cell-mean全部可观测未来SOH的MAE/RMSE，保留四个时距区间、协议宏平均与unseen-protocol结果。与已完成M4进行配对cell比较，遵守原split和观测mask。A1额外报告回升诊断。

**优势**是每项以同一M4为基准，机制替换范围明确，数据、训练预算与选择规则共同固定，可追踪各机制的单独预测作用。

**风险**包括单seed训练随机性未覆盖、结构性参数量差异、A4仍保留raw VIQ的局部信息、A1放松单调约束后的轨迹回升，以及169电芯已有历史测试暴露。结果属于原任务内部的探索性机制比较。完整重复训练及首批位级复现的支持范围依据实际完成的收据披露。

结论置信度依据实际实现QA、主矩阵完成数与重放证据分项登记。合同本身只定义本轮实验，模型效果结论由真实输出决定。
