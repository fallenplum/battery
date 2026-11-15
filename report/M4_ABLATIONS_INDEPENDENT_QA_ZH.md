# M4四项独立消融实现审查

## 实际执行结果

**CPU231项、CUDA271项独立工程检查全部通过。** 每个arm在CUDA上使用真实冻结train电芯首批、四线程train/dropout、全部未来query及原full-loss，fresh seed42重复12次，预测、loss与全部参数梯度hash逐次完全一致。工程控制的optimizer updates为0，test评分次数为0。

四项分别以M4为直接基准，保留的同shape张量逐值核初始化，结构性变化限定到各自指定模块。旧七臂43份注册源码和M4八份注册依赖逐文件SHA保持，原冻结loader与169电芯数据、标签、split指纹通过。

权威收据为`results/unified_matr_ablations_seed42/INDEPENDENT_ABLATION_QA_CPU_FINAL.json`及同目录`INDEPENDENT_ABLATION_QA_CUDA_FINAL.json`。入口为`scripts/verify_m4_ablation_contract.py`。

## 逐项信息与计算路径验证

| Arm | 被替换机制的实际检查 | 保留路径的实际检查 |
|---|---|---|
| A1_M4_LEGACY_FLOW | 与原LegacyFlowHead同初始化、同state/condition的解析预测逐值一致，signed loss为负correction，t>100的signed rate与autograd导数一致 | state、condition及全部共享编码路径与M4逐值一致，B写入在受控W_B扰动下继续进入真实SSD |
| A2_M4_NO_B_WRITE | 捕获两层SSD输入，均为真实B且Bprime=B，gate/W_B参数移除 | 初始预测与零初始化W_B的M4逐值一致，动态rho、quality、持久state与M4一致，selector_condition受控移除会改变正常预测 |
| A3_M4_STATIC_RHO | rho仅64个可学习logits，初值0.9，响应、质量、policy扰动后rho不变，logits获得非零梯度 | 实际gap仍取指数，首有效圈初始化与缺失持有状态保留，B写入、selector、decoder及其他路径保留 |
| A4_M4_NO_LOCAL_PHYSICS | local值置NaN且mask全部翻转，预测、trend及penalty完全不变，local residual/features梯度为空或零，quality32全部为零 | raw/reference VIQ、global20、policy逐项合法扰动均改变正常预测，局部self-reference Mamba保留，共享参数保持原初值 |

每个arm另验证leading缺失后的first-valid初始化、invalid记录持有内部状态、五类输入的invalid记录梯度为0、前缀后部扰动不改变早期rho/state/B、cycle101不能作为prefix输入、100圈anchor、观测残差0.03界限及query单独/联合/乱序一致。A2/A3/A4保留非负rate和单调trend结构。A1采用允许回升的原Flow，受控反例产生SOH回升0.02006748，保留该结构事实。

M4原动态EMA及SSD另以独立逐时标量递推核值和梯度。A1的100圈anchor走原`torch.where(tau==0)`分支，导数比较使用t>100，避免将anchor分支的点导数与Flow右导数混用。

## 参数量与边界

| 模型 | 参数总数 | 相对M4变化 | 受控支路激活下有非零梯度的参数元素 |
|---|---|---|---|
| M4_FULL_METHOD | 154621 | 0 | 原M4收据记录 |
| A1_M4_LEGACY_FLOW | 154621 | 0 | 154621 |
| A2_M4_NO_B_WRITE | 149469 | -5152 | 149469 |
| A3_M4_STATIC_RHO | 139453 | -15168 | 139453 |
| A4_M4_NO_LOCAL_PHYSICS | 135245 | -19376 | 135245 |

表中梯度计数来自固定手工W_B扰动后的工程反向控制，用于检查保留参数可达性。新模型的原始W_B仍按合同零初始化。参数量差异来自被移除机制及连接，未加入参数匹配模块。A4仍保留raw VIQ的局部曲线信息，结果解释范围限定为显式局部物理特征支路。

A4的global combine由128输入保留64个global列，rho移除原context列104至135，实际保留140列，gate保留persistent-state64列。相应权重与bias逐值继承M4。内部context布局可继续形成172/96维张量，由ActiveColumnsLinear读取保留的140/64列。注册维度描述对应可训练连接。

## A4跨矩阵形状的舍入诊断

首轮CUDA检查使用FP32比较128输入的M4零local token与64输入的裁列实现，最大差7.152557373046875e-7，超过当时预设5e-7。失败原件保留为`INDEPENDENT_ABLATION_QA_CUDA_FP32_COMPARISON_ATTEMPT1.json`。

两边权重保留列和bias逐值相同，CPU该项差值为0。独立CUDA FP64全global物理路径oracle差值为0，满足2e-12阈值。最终检查使用FP64严格验证数学等价，同时记录FP32跨GEMM形状的舍入差。该测试修正由主执行者明确确认，模型与runner源码保持原SHA，同一arm的12次首批bitwise标准继续使用完全一致要求。

保留的CPU中间收据包括初版222项及补充first-valid控制后的230项，最终CPU/CUDA收据绑定同一份最终QA脚本。

## Runner只读审查

共同loss直接引用M4原`training_loss`，rate辅助仍取预测SOH端点差，fold-local scaler复用原函数。smoke针对A2正确识别无W_B参数，其他三项验证首批W_B梯度。每项smoke仅一epoch，正式参数重新seed42初始化。

正式执行顺序为arm-first的24fits，共49800updates。四项fold0完整重复安排在主矩阵全部完成之后，共8400updates。四次smoke共56updates，实际计划总数58256。严格first-min val选择、150epochs、checkpoint冻结后test评分与M4一致。M4结果通过已完成且验证的保存产物引用。

## 优势与风险

**优势**是按共享初始化、被移除输入的零作用、保留输入的实际作用、真实SSD输入捕获与完整首批梯度hash交叉验证各项消融范围。

**风险**包括工程检查只覆盖所执行控制和首批数据，完整150epoch复现仍依赖后续实际训练。单seed未覆盖跨训练seed差异，结构性参数量变化与信息移除相伴。169电芯及M4既有结果已暴露，后续比较继续按固定数据内部探索性证据报告。

置信度评级为**高**，限本报告实际执行且SHA绑定的合同实现与数值控制。

## 最终版本绑定

| 文件 | SHA256 |
|---|---|
| research/M4_ABLATIONS_SEED42_CONTRACT_ZH.md | 8c67839cb66210f6bb5117a6510035f8773d3abbea0a17a60d08d16c04a508dc |
| models/unified_matr/methodology_ablations.py | 8ad6ac6d7b67f103baece5b88d38a2c8719778cbb6577d4e9b51a6f171490330 |
| scripts/unified_matr/train_methodology_ablations.py | 085e342209122b4c6fab51507f8cf13d712ae787626bc8484a548b5cdd3e6a7e |
| configs/unified_matr_ablations_seed42.json | 564cd5bdcf2ff8283247b4f26b669f1ad0f5bd0cd52641ef4d8c1050a80344d8 |
| scripts/verify_m4_ablation_contract.py | db76d8309469598dc04c5ad19b1ebc709fc7d4937d5ee4407afbfaa5a1417726 |
