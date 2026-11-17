# 共同源域Bridge/Head下的主干专门化与目标少样本适配

## 研究问题

本轮比较官方预训练主干配合同一源域Bridge/Head，与增加源域静态核心后的参考函数，判断源域专门化怎样影响目标域预测及少样本校准。随后在同一专门化参考函数上比较标准LoRA、LoRA-XS零增量版本和OUT16。

执行配置位于configs/drs_peft/specialization_comparison_v1/CONFIG.json。父实验执行提交为c04bea3a8313b17df3676c3d3b9c00623bfd2f75。新的实验、输出和报告使用specialization_comparison_v1独立目录。

## 共同输入与两类参考函数

输入沿用VIQ+earlySOH / no-policy。前100原始循环、合法前缀mask、固定循环坐标及已有earlySOH锚点构成模型可用信息。native_multivariate消费每圈12个统计描述量。policy维度保持0，容量单位与额定容量分母沿用父实验。未来圈号固定为101至5000，真实Query观测mask只用于评分。

设官方预训练权重为W₀，同一fold、seed的源域Bridge/Head为θBH。warmup只训练59585个Bridge/Head参数。static阶段冻结θBH，只拟合12个rank-4源核心，共192个参数。每个线性模块的专门化有效权重记为

$$
W_s=W_0+s_0U_0C_0V_0^\top.
$$

式中使用实际保存的源核心、基和缩放。两类参考函数分别为

$$
f_w(x,t)=f(W_0,\theta_{BH};x,t),\qquad
f_s(x,t)=f(W_s,\theta_{BH};x,t).
$$

WARMUP_ZERO使用f_w，SPECIALIZED_ZERO使用f_s。warmup的Bridge/Head已经利用源电池监督训练。该命名标识官方预训练主干及共同源接口的组合。

复用父实验的9组源CV参考与3组最终源参考。最终参考仅使用原115颗源电池，训练epoch预算沿用源CV中位数规则。每组warmup和static的Bridge/Head权重必须逐项一致，checkpoint身份分别由warmup_checkpoint_sha256和reference_checkpoint_sha256登记。

## 四种目标适配

| 方法 | 参考函数 | 更新形式 | 目标可训练参数 |
| --- | --- | --- | --- |
| PRETRAINED_LORA | f_w | 新增BA，rank1 | 9216 |
| SPECIALIZED_LORA | f_s | 保留源核心并新增BA，rank1 | 9216 |
| SPECIALIZED_XS | f_s | 新增UᵣδRΣᵣVᵣᵀ，每模块rank4 | 192 |
| SPECIALIZED_OUT | f_s | 原OUT字典的跨模块共享a₁₆ | 16 |

四种方法均冻结Bridge、Head、官方基础参数及既有源核心，只优化表中新增适配参数。新增分支的函数增量从零开始。SPECIALIZED_LORA、SPECIALIZED_XS和SPECIALIZED_OUT在K0共享SPECIALIZED_ZERO。

两种LoRA都采用随机Kaiming A、零B、α=r=1、scale1、dropout0。隔离随机种子及模块编号规则沿用父实验，保证同seed在不同K、draw和超参试验中的初始化一致。PRETRAINED_LORA与SPECIALIZED_LORA的相同目标参数预算用于比较源域专门化的影响。

## LoRA-XS的数学身份

LoRA-XS使用权重SVD给出冻结左右因子，只优化r×r中间矩阵。官方代码对转置权重做SVD，PyTorch的W∈Rᵐˣⁿ方向下，新增矩阵为sUᵣRΣᵣVᵣᵀ。奇异值Σ影响优化坐标，属于该实现的组成部分。[LoRA-XS论文](https://arxiv.org/html/2405.17604v3)、[官方SVD实现](https://github.com/MohammadrezaBanaei/LoRA-XS/blob/main/utils/svd_utils.py)、[官方前向](https://github.com/MohammadrezaBanaei/LoRA-XS/blob/main/utils/latent_utils.py)

本轮对每个模块的完整专门化有效权重W_s做CPU FP64完整SVD，取前4个奇异方向，冻结Bₓ=Uᵣ及Aₓ=ΣᵣVᵣᵀ，并附加

$$
\Delta W_{XS}=\eta\frac\alpha r U_r\delta R\Sigma_rV_r^\top,
\qquad r=\alpha=4,\quad\delta R_{init}=0.
$$

12个线性模块的δR共192个参数。原W₀及已拟合源核心继续参与前向，SVD基来源为二者相加后的有效W_s。缩放η作用于整个新增增量。

官方初始化给R使用标准差10⁻⁵的Gaussian分布。本轮为了精确复用共同f_s，采用δR零初始化，并登记为**LoRA-XS zero-increment variant**。数值SVD采用确定性的完整分解，官方实现使用随机TruncatedSVD。这两项实现选择写入配置和模型回执。[官方初始化](https://github.com/MohammadrezaBanaei/LoRA-XS/blob/main/utils/initialization_utils.py)

本地models/drs_peft/peft.py中的LoRAXSLinear保留Σ，并使用Gaussian核心。新实现需要按上述零增量及W_s构基定义接入。实际冻结因子、奇异值buffer及新增核心的字节数在预算表中分别计量。

## Support与Query身份

复用configs/drs_peft/onesided_fewshot_v1/EPISODES.json，其合同hash为384e9251ab3522372d675d4c352a105688ebf851c8b0b19dc1e13e627d08472e。物理电芯身份、Support池、固定Query、K与draw逐项继承。

MATR主评价为4个精确工况任务，共8颗固定Query。每任务分别获得K颗Support，整个MATR聚合信息预算为4K。HUST使用固定39颗Query及38颗Support候选池，每次从池中取得K颗。

K为0/1/3/5，非零K各5次既有嵌套draw，训练seed为42/123/2026。相同Support集合允许复用相同适配状态，全部预登记draw仍进入结果。每个参考family、task和seed只登记一次K0、draw0。

目标完整矩阵为900个非零请求槽和30个family ZERO参考槽。900个请求槽对应732个唯一非零状态，每个方法183个。PRETRAINED_LORA与SPECIALIZED_XS合计新增366个唯一非零状态，另外新增15个WARMUP_ZERO参考槽。SPECIALIZED_LORA、SPECIALIZED_OUT及SPECIALIZED_ZERO复用父实验已经锁定的目标预测，其checkpoint、输入、episode、超参数及预测身份须一致。

目标Query身份已被历史实验复用，本轮继续登记其暴露状态。结果用于同Query配对比较。Stanford不进入本轮目标矩阵。

## 源内部选参与适配预算

SPECIALIZED_LORA和SPECIALIZED_OUT直接继承父实验TARGET_LOCK.json中的LoRA与OUT超参数，其源inner试验及既有选择记录保持。

仅PRETRAINED_LORA与SPECIALIZED_XS新增source-inner选参。三折inner任务各8颗固定Query，Support池为7/7/8颗。每个K使用同一组五次draw，在学习率10⁻⁴/10⁻³、步数10/30、η为0.25/0.5/1的12项组合中选择。优化器为AdamW、weight_decay0、gradient_clip1，Support损失按物理电芯等权的观测均方误差计算。

每颗Support最多取512个确定性等间隔真实观测点。学习率、步数与η仅由源inner Query的完整未来MAE决定。对每个方法及K，将三折、三seed、五draw等权汇总，选择第一个严格最优组合，随后在所有目标域、任务、seed和draw上共用。试验次序为学习率升序、步数升序、η升序。

新增两方法的source-inner共3240个评分槽。优化轨迹及中间步数快照可以缓存复用，实际优化步数和缓存命中另外记账。新增source-outer实验数为0。全部新增方法的选择、SVD因子、初始化及目标超参数锁定后，生成固定网格预测，再执行评分。

## 指标与配对差异

每颗Query在其实际观测未来上计算MAE，再对电芯平均。完整未来与101–200、201–500、501–1000、1001+分段共同报告。无观测窗口保留NA，同时报告有效电芯、工况与观测点数。

每种方法的适配收益对应其参考family

$$
\Delta_K^{(m)}=MAE(f_{family(m)})-MAE(f_{m,K}).
$$

Δ_K正值表示同一family内的适配改善。另报告两个零样本family的MAE差，以及PRETRAINED_LORA与SPECIALIZED_LORA的绝对MAE差，用以保留源专门化带来的基线位移。

OUT相对专门化LoRA和XS的主差异为

$$
d_{OUT-LoRA}=MAE(SPECIALIZED\_OUT)-MAE(SPECIALIZED\_LORA),
$$
$$
d_{OUT-XS}=MAE(SPECIALIZED\_OUT)-MAE(SPECIALIZED\_XS).
$$

两个差异均以负值支持OUT。按相同domain、task、seed、K、draw、物理电芯和window配对。各方法负迁移的基准为自己的family ZERO，seed波动与Support抽样波动分别报告，新增bootstrap次数为0。

## 优势、风险与结论范围

共同Bridge/Head、相同输入单位、固定Query及继承Support抽样有利于识别源核心与目标适配的实际作用。PRETRAINED_LORA和SPECIALIZED_LORA共享目标参数规模和优化网格，支持对源专门化影响的受控比较。

XS与OUT分别使用192和16个目标参数，同时具有不同构基与求解方式。该比较衡量实际预测效果与成本，单一优势无法隔离所有结构因素。XS的零初始化、专门化权重构基和完整SVD需要在论文身份中保留。目标Query的历史暴露、有限工况数、远期观测不足及源inner的历史使用限制独立性。

结果置信度由实际评分覆盖、三seed方向、Support抽样波动和跨域一致性决定。完整轨迹、远期损失、负迁移和成本共同构成结论。所有方法、family和失败状态保留在汇总中。

## 交付与工程核验

公开结果位于results/drs_peft/specialization_comparison_v1，私有预测与适配状态位于runtime对应目录，报告位于report对应目录。交付CELL_METRICS.csv、ADAPTATION_RECORDS.jsonl、REFERENCE_FAMILIES.csv、TARGET_LOCK.json、ADAPTER_BUDGETS.csv、TARGET_FEWSHOT_RESULTS.csv、PAIRED_METHOD_DIFFERENCES.csv、RUN_LEDGER.json及中文复核。

核验覆盖两个family参考重放、Bridge/Head权重一致、XS有效W及Σ方向、零增量重放、目标可训练参数作用域、同K与draw身份、保存加载重放和真实非线性评分。Query未来标签只进入评分函数，Support未来监督仅限当次抽样电芯。

## 拓展思考

结合两个family的零样本位移与各自Δ_K，可以分析专门化主要改变了源到目标迁移的起点，还是改变了目标Support利用效果。

## 你可能想知道

共享Bridge/Head包含源电池监督信息。比较中的额外源域专门化来自冻结后的192参数静态核心，目标域适配再使用当次K颗物理Support。
