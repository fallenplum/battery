# Battery：总 K 预算下的校准与迁移范围实验计划（v2）

## 0. 研究定位、起点与执行范围

审查仓库：fallenplum/battery。
审查分支：codex/fewshot-one-sided-transfer-v1。
审查提交：49fd7c210b60297bcf38daaccd7535ebabf32d35。

这是下一轮研究计划，不是已经得到验证的新算法。审查依据是公开报告、配置、指标 CSV 与关键模型代码；未在本地重训模型或独立重算全部私有预测 NPZ。

核心问题：当目标域只有 K 颗完整未来轨迹标注电池时，如何区分可共享的整体轨迹校准与电芯特有的修正，避免支持集拟合改善却导致其他 Query 电芯负迁移？

本轮不扩大 OUT 秩、不新训练基础模型、不恢复双视图门控、不重新运行 ABC/E123 全矩阵。复用 no-policy 源 checkpoint、OUT 字典、合法前缀、Support/Query 身份和 Jacobian 缓存。主要新增工作是小矩阵求解、必要的非线性前向与局部适配对照。

建议新分支：codex/transfer-scope-calibration-v2。检查工作区后从上述提交建立；保护未提交修改和所有历史结果。不要覆盖旧配置或修改旧报告。

## 1. 必读文件

- report/drs_peft/onesided_fewshot_v1/FINAL_REVIEW_ZH.md
- report/drs_peft/specialization_comparison_v1/target_total_K/FINAL_REVIEW_ZH.md
- report/drs_peft/specialization_comparison_v1/target_total_K/ADAPTATION_DIAGNOSTICS_ZH.md
- configs/drs_peft/onesided_fewshot_v1/CONFIG.json
- configs/drs_peft/specialization_comparison_v1/CONFIG.json
- configs/drs_peft/specialization_comparison_v1/TARGET_TOTAL_K.json
- configs/drs_peft/specialization_comparison_v1/EPISODES_TOTAL_K.json
- research/drs_peft/DATA_SPLIT_LOCK.json
- results/drs_peft/specialization_comparison_v1/target_total_K/TARGET_FEWSHOT_RESULTS.csv
- models/drs_peft/onesided.py
- models/drs_peft/specialization.py
- models/drs_peft/bridges.py
- models/drs_peft/trajectory.py
- 对应 run_specialization、源 inner 选参与 target_total_K 执行入口；通过仓库目录找到真实文件，不猜造命令。

## 2. 本轮必须承接的事实

### 2.1 两种 K 不能混报

旧 MATR：4 个精确协议任务各使用 K 颗 Support，合计 4K，分别适配多个状态。
新 MATR：从 24 颗候选中全域总共使用 K 颗 Support，共同适配后评价固定 8 颗 Query。
HUST：从 38 颗候选中全域总共使用 K 颗 Support，评价固定 39 颗 Query。

原 MATR OUT K5=0.014551 对应 20 颗总 Support 和协议任务分开适配；新总 K5=0.015761 对应 5 颗总 Support 和共享状态。两者不能用于单因素因果比较，也不能写成同一 5-shot 方法结果。

### 2.2 总 K 主结果

| 域 | 方法 | K0 | K5 | 说明 |
|---|---|---:|---:|---|
| MATR | WARMUP+LoRA | 0.016023 | 0.016441 | 适配恶化 |
| MATR | SPECIALIZED+LoRA | 0.015413 | 0.016139 | 适配恶化 |
| MATR | SPECIALIZED+XS | 0.015413 | 0.015489 | 接近不适配，但仍恶化 |
| MATR | SPECIALIZED+OUT | 0.015413 | 0.015761 | 适配恶化 |
| HUST | WARMUP+LoRA | 0.035917 | 0.032048 | 适配改善 |
| HUST | SPECIALIZED+LoRA | 0.036558 | 0.032900 | 适配改善 |
| HUST | SPECIALIZED+XS | 0.036558 | 0.036453 | 小幅改善 |
| HUST | SPECIALIZED+OUT | 0.036558 | 0.034953 | 改善但不及 LoRA |

已有 HUST CAL2 K5=0.029411，在相同父协议与参考函数下优于上述复杂适配；复用前逐项确认 SHA、Support、Query、源权重和超参数一致。新 MATR pooled-K 尚缺这一基线，必须补齐。

360 个非零记录 Support MSE 全部下降；MATR 的 180 个记录中有 130 个 Query 平均 MAE 恶化。这些是重复 seed/draw 的记录数，不是 180 颗独立电池。

### 2.3 已执行而不应重复的工作

- D0–D4 已完成，真实目标适配不是未运行。
- 主输入已是 VIQ+earlySOH / no-policy。
- 当前 OUT 由 12 层共享同一 16 维系数；每个 episode 中所有 Query 复用该系数。
- 同 checkpoint 构基与实现重放已有记录。
- CROSS 当前未白化、一次局部求解版本未改善；不把整个跨电芯理论否定，也不再扩大该搜索。
- XS 已证实保留 Sigma、有实际权重更新；不要再用“XS 没有更新”解释其弱收益。
- WARMUP 的 Bridge/Head 已接受源监督；不能把它称为未经电池学习的原生零样本 TSFM。

## 3. 文献依据与新颖性边界

执行前阅读并记录原论文相关部分，而不是只复制二手摘要：

1. Wortsman et al. Robust Fine-Tuning of Zero-Shot Models. CVPR 2022.
   https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html
   借鉴源权重与微调权重之间的收缩；原论文并不证明电池场景有效。
2. Harrison, Sharma, Pavone. Meta-Learning Priors for Efficient Online Bayesian Regression (ALPaCA).
   https://arxiv.org/abs/1807.08912
   借鉴源任务学习先验、小样本下低维正则化更新。
3. Patacchiola et al. Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels. NeurIPS 2020.
   https://proceedings.neurips.cc/paper/2020/hash/b9cfe8b6042cf759dc4c0cccb27a6737-Abstract.html
   借鉴基于输入相似性传递支持信息；局部核与少样本贝叶斯预测本身不是新颖点。
4. Zhu, Cui, Zhang. Robust Fine-tuning of Zero-shot Models via Variance Reduction. NeurIPS 2024.
   https://proceedings.neurips.cc/paper_files/paper/2024/hash/8cc7e1509fbfee9cabaacd3ab0bfe2b1-Abstract-Conference.html
   查询样本相关的融合已有先例；不能把“相似度门控”本身宣称首次提出。

本轮首先验证机制，不预写新算法优势，不宣称“电化学机制识别”。条件加权岭回归、Schur 补、收缩均为既有数学工具。论文潜在贡献必须建立在电池特有的响应—退化关系、合法少样本协议与真实跨电芯收益上。

## 4. 三个研究假设

H1：整体轨迹尺度/速率校准可解释一部分目标偏移，复杂 TSFM 更新应证明超出 CAL2 的增量。
H2：不同电芯未必适合同一个参数修正；早期响应相似的 Support 对某颗 Query 的修正更可迁移。
H3：支持数据与 Query 关联弱时应缩小更新，而不是被迫应用至少 0.25 倍的更新。

现有结果只支持检验这些假设，尚未确认其因果真实性。没有任何规则按 MATR/HUST 名称决定更新强度；两域使用同一源内锁定算法。

## 5. 固定信息与评估合同

- 每个 Query 仅用前 100 圈 VIQ、合法 earlySOH/anchor、前缀 mask 和原循环坐标。
- 不输入 policy、policy_mask、协议 ID、dataset ID、温度、材料、未来工况、寿命长度或未来 mask。
- 支持未来标签限当次选中的 K 颗电池。相似性来自每颗 Query 自己的前缀，不聚合其他 Query 的统计量。
- 两域统一为总 K∈{0,1,3,5}。同 draw 的支持集嵌套，同方法共享身份，同 K 固定 Query。
- 复用原 5 个 draw，保留重复标记；MATR K1 只有 3 个不同集合，不写成 5 次独立抽样。不需要为本轮重抽全部划分。
- 预测网格固定 101–5000；评分只用真实观测位置。每个电芯先平均，再跨电芯平均。
- 最多使用每 Support 512 个确定性实际观测点，沿用旧规则。不能把这些点称为独立 shot。
- 主评估不是 EOL；HUST 的删失范围与各时距实际支持必须明确。
- 目标 Query 已暴露，本轮为回顾性受控比较，不再称独立盲测。
- 新的正则、核参数、收缩参数仅在源 inner 任务选取，不能根据目标结果重新调参。

## 6. 共同参考与复用范围

主机制矩阵使用 SPECIALIZED f0，以复用已有 OUT16 字典、源梯度、JVP 和全部目标参考，且保证 CAL2 与内部更新是同起点比较。

保留 WARMUP+LoRA 和 SPECIALIZED+LoRA 的原锁定结果作为实际竞争锚点。两者不同 K0 必须分别列出；不能以两家族绝对差异断言 source specialization 普遍有害，已有均值明显受 seed123 影响。

新源 checkpoint 拟合预算为 0。必要的新特征统计或小型 inner 选择单独记账。禁止把再次训练 Bridge/Head 隐藏在“适配”里。

## 7. E0：补齐比较与收缩对照

### 7.1 补齐 CAL2

在新的 MATR total-K 身份上补跑 SPECIALIZED+CAL2；HUST 仅在完全同身份时复用。

CAL2 采用：

  yhat_i(t)=f0_i(t)+c1*(f0_i(t)-anchor_i)+c2*(t-100)/1000。

保留 t=100 的锚点；无单调性强制、不依赖真实 EOL。使用与后续矩阵相同的 source-inner 正则/收缩选择预算。

### 7.2 合法的 zero-retaining 对照

旧 eta 网格为 {0.25,0.5,1}。新网格固定为 {0,0.1,0.25,0.5,1}。

- 对已保存的、原源内选定学习率和步数所产生的 LoRA 更新，只改变整体增量缩放。不重新搜索学习率/步数。
- 用源 inner 重新选 eta，目标不参与。源内选择 eta=0 合法，记为“不适配”；不要用目标 ZERO 优于适配的事实直接按域选择 eta=0。
- 同时保留旧非零策略和新可拒绝适配策略。不能仅因输出全部退回 ZERO 就宣称新方法成功。
- 若旧状态只保存已缩放因子，必须依据保存格式恢复未缩放增量；不能猜测 A、B 同时缩放，否则会把 eta 错写为 eta²。

## 8. E1：固定前缀相似性，不新增大网络

主表示 z_i 使用共同 f0 的冻结 64 维 encode 输出，只依赖合法前缀；不得使用未来残差、EOL 或已知协议作为 z_i。

标准化参数仅由该 fold 的 fit 电芯确定；final-source 版本仅由原 115 源电芯确定。去掉零方差维，不用目标 Query 全体重新标准化。不重新学习编码器或新增门控 MLP。

定义距离：

  d(i,q)^2 = mean_j [(z_ij-z_qj)/s_j]^2。

s_j 使用源 fit 标准差与已声明数值下限。设源 fit 不同电芯之间正距离的中位数为 d0；退化到全相同表示时局部方法标记无辨别性，并退回可复现的全局权重，不伪造局部有效性。

主核：

  w_iq = exp[-d(i,q)^2/(2*tau^2)]，tau/d0∈{0.5,1,2}。

全局控制 w_iq=1。不要对 w 做 softmax 或除以 sum(w)：当所有支持电芯都远离 Query 时，应允许绝对权重整体减小，从而保留源模型。

正则尺度必须使用 source-fit 固定尺度。禁止以当前加权 Gram 的 trace 重新设定 lambda；那会使 w 整体缩小后正则也同比缩小，抵消“远离支持就收缩”的机制。

记录每个 Query 的 sum(w)/K、最大 w、有效邻居数 (sum w)^2/sum(w²)；这些是支持覆盖诊断，不是已校准置信概率。

## 9. E2：2×2 主矩阵及既有参考

主矩阵：

| 方法 | Support影响范围 | 校准空间 |
|---|---|---|
| G-CAL2 | 全部 Query 共享 | CAL2，2维 |
| L-CAL2 | Query前缀相关 | CAL2，2维 |
| G-JOINT18 | 全部 Query 共享 | CAL2 + OUT16，18维 |
| L-JOINT18 | Query前缀相关 | CAL2 + OUT16，18维 |

另保留 ZERO、旧 G-OUT16、两种家族的原标准 LoRA；零保留 LoRA 收缩对照按 E0 执行。四个主矩阵方法完整运行，不因某一个源域收益小而跳过其他对照。

这些比较分别回答：
- L-CAL2 vs G-CAL2：只改变信息共享范围是否足够？
- G-JOINT18 vs G-CAL2：共享设置下内部微调是否有额外信息？
- L-JOINT18 vs L-CAL2：局部设置下内部微调是否有独立价值？
- L-JOINT18 vs G-JOINT18：条件化信息传递是否改善少样本迁移？
- 所有方法 vs ZERO/LoRA：不是仅击败较弱的 OUT 参考。

### 9.1 线性化与小矩阵求解

对 Support i 的实际采样时点，缓存：
- e_i = y_i - f0_i；
- B_i 两列分别为 f0_i-anchor_i 与 (t-100)/1000；
- J_i = d f_i / d a，a为已有 OUT16 的零点坐标。

设 x=[c;a]，A_i=[B_i,J_i]。求解：

  min_x  (1/K) sum_i (w_iq/n_i)||e_i-A_i x||² + xᵀ R x。

G 方法的 w=1；L 方法使用 Query 前缀核。CAL2 方法只保留前两列。

R=diag(lambda_c*s_c*I2, lambda_a*s_a*I16)。s_c、s_a 为源 fit 中对应平均 Gram 的 trace/dimension，并在进入 inner/target 前冻结。主网格以共同 lambda multiplier∈{0.1,1,10}同时作用两块；不在第一轮自由扩成 3×3 两块参数搜索。eta∈{0,0.1,0.25,0.5,1}整体收缩解。

采用 CPU float64 Cholesky/稳定线性求解，不显式求逆。所有 ridge 值严格正，eta=0提供合法 no-op。

不要将局部加权平方损失的形式擅称已校准贝叶斯后验；当前仅是带正则的局部代理目标。

### 9.2 Schur补与残差解释

小矩阵可分块：Hcc, Hca, Hac, Haa；右端bc,ba。

  a=(Haa-Hac*solve(Hcc,Hca))^{-1}(ba-Hac*solve(Hcc,bc))，
  c=solve(Hcc,bc-Hca*a)。

实现使用solve。这使 OUT 的解考虑已被 CAL2 解释的部分，避免将“校准水平/速率”与“内部表征修正”混为同一贡献。

有 ridge 时是带收缩的残差化，不是严格正交投影。Schur补本身不作为算法新颖性声明。

### 9.3 真实非线性评分

最终预测固定为：

  yhat_q(t)=f_{theta0+DeltaW(a_q)}(X_q,t)
            +c_q1*(f0_q(t)-anchor_q)+c_q2*(t-100)/1000。

CAL2 项基于原 f0，不对已经更新的 f 再乘 gamma，避免引入未在代理系统中声明的乘性交叉项。f0预测可复用缓存。t=100下两个新增项都为0，内部头的原锚点保持结构不变。

不能用 f0+J*a 替代非线性模型主预测。记录线性化误差，分别列实际观测范围和全固定网格，不将未观测远期差异解释为真实预测误差。

L 方法使用同一批 K 个已标注支持电芯为每颗 Query 求出不同系数，并没有增加标签预算。但每个 Query 有独立的2/18维有效系数，不能宣称整个目标域只有2/18个共享参数；报告总适配状态数、求解次数和推理耗时。缓存键必须包含Query前缀身份或等价权重指纹。

## 10. 源内选择、预算与执行顺序

1. 先完成静态文件/身份复用检查与少数真实样本 smoke，不重复原始数据全量审计。
2. 复用全部共同参考模型；源预训练次数为0。
3. 在现有源 inner 支持/查询任务上，计算或复用 B、J、f0、前缀z。以支持MSE为代理求解，但选参和主指标使用Query完整已观测轨迹的电芯均值MAE。
4. 局部候选最多 3 lambda ×3 tau ×5 eta；全局最多3 lambda ×5 eta。允许先用缓存线性化预测筛选，再对预定前3名加ZERO进行真实非线性inner评分；所有几何方法用同一规则，并记录筛选方式，不能把近似选参描述为全量非线性搜索。
5. 选择按每方法、每K聚合源三折三seed五draw；目标域不可作为选择特征。相同分数时优先小更新、强正则，规则运行前固定。
6. 方法、参数和代码锁定后，一次执行两域所有主矩阵方法和K。不要查看一个目标的结果后修改另一个目标的算法。
7. 目标真实数据只有当前K颗Support用于拟合，Query标签只用于评分。Query前缀可计算自己的权重，其他Query不得参与转导学习。
8. 每个技术失败局部处理并留日志，不因为统计门槛、置信区间或小于1%收益跳过整个矩阵。

本轮不做 bootstrap 显著性门槛、不要求所有fold/seed/窗口同时获胜；报告原始效果量、逐seed/逐draw/逐电芯结果和负迁移即可。

## 11. 可选扩展评价：增加电芯，而不是增加重复行

身份复核原 MATR target54。如果24颗固定Support候选的补集确有30颗输入合格且未参与源训练的电芯，可在所有方法锁定后新增扩展Query评价：
- 主结果仍为旧8颗，保证历史可比；
- 新增22颗单独报告；
- 扩展总30颗另列。

上述数字须由UID集合实际复核，不能仅凭54-24推断资格。不得按预测误差、寿命长度或方法收益筛选Query。源历史研究已使用过这些电池，因此不能称为全新独立外部证据；本轮首次评价状态与历史暴露分开登记。新增精确工况覆盖不同，需要单列。

HUST保持固定39 Query和38 Support，不更换以获得更好结果。

## 12. 最小核验

- K=0和eta=0恢复各自f0；新增加载、合并、参考指纹不改变起点。
- 所有方法总Support数严格为K，无4K混用。
- Query后缀扰动不影响其输入、相似权重、适配系数、预测。
- permute Support顺序不改变结果；重复整块电芯不算新shot。
- w=1的局部求解恢复全局解；w趋0且R固定时解趋0。
- CAL2与OUT列置零分别恢复对应单模块；联合18维直接solve与Schur解一致。
- 同query同support同配置重放，数值容差与dtype匹配即可，不重新要求跨平台逐bit训练相同。
- nonlinear评分确实使用适配模型；不是只输出缓存线性预测。
- 新增标签可见性限制和no-policy从源至目标保持。
- 参数、字典存储、源选择和目标求解开销分别报告。

## 13. 报告与文件交付

建议独立命名空间：configs/research/results/report 下的 transfer_scope_v2。

至少交付：
- CONFIG.json：完整输入合同、方法矩阵、网格和选择规则。
- REUSE_MANIFEST.json：复用checkpoint/字典/缓存身份与缺失情况。
- SOURCE_SELECTION.json：仅源inner的参数选择结果。
- TARGET_TOTAL_K_LOCK.json：总K身份、Query列表、目标开封前锁。
- METRICS.csv：domain/family/method/K/window/seed/draw/cell及聚合误差。
- TRANSFER_DIAGNOSTICS.csv：距离、绝对权重、有效邻居数、系数范数、实际预测位移、Support拟合与Query收益。
- CAL2_INCREMENTAL.csv：联合方法相对CAL2的额外收益。
- EXECUTION_LEDGER.json：训练/求解/前向/缓存命中分别计数。
- FINAL_REVIEW_ZH.md：正负结果与失败个体、各来源边界。
- 可重放状态、必要预测和一条主执行入口；主命令必须真实运行，不能只留伪代码。

主要表同时列：绝对MAE、自己的K0、相对K0改善、相对CAL2与LoRA的差异、时距分母、参数/状态预算。三seed与五draw不当作独立电芯数量。

## 14. 结果驱动的研究决策

- L-CAL2明显改善而L-JOINT18没有额外收益：当前问题主要是校准和迁移范围，停止把OUT视为论文核心；结果不证明内部微调永远无效。
- L-JOINT18在固定总K下稳定优于L-CAL2、G-JOINT18和标准LoRA：才继续发展“响应相关的结构化PEFT”，并开展新队列/新支持组合验证。
- eta=0频繁入选：如实报告需要适配的情形有限；不能将不适配本身包装为算法增益。
- 所有局部方法更差：现有前缀表示或核距离缺乏迁移信息，不能直接证明局部适配概念错误。此时只做有限、源数据内的曲线形态保留接口对照，不再调大门控。
- 支持误差下降但Query持续恶化：优先分析支持代表性、跨电芯差异与收缩，而不是继续追求更低的Support loss。

严格禁止按目标域名字硬编码“一个不调、另一个微调”；也禁止看目标结果挑选某个seed、draw、K或时距后作为主结果。
