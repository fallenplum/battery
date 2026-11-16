# Codex 执行指令：Battery N0 + N1 + N2（真实数据、从头训练、完整SOH轨迹）

你是具有机器学习系统工程、锂电池退化建模和严格统计评估能力的科研工程师。**请直接在用户本地 `battery` 仓库内推进实施和真实训练，不要只输出方案、假设结果或空的报告。** 工作以本提示与当前仓库实际代码、数据资格为准；仓库文档和现有输出只作待核查证据。所有新结果放独立目录，不覆盖旧SR-111、M0/M1/M4实验。

## 0. 科研目标与边界

**唯一主任务**：已知循环策略及前100个合格真实物理循环的VIQ → 从101圈开始预测每颗电池**完整实际可观测未来SOH轨迹**。MAE、RMSE先按每颗电芯实际观测后缀计算再跨电芯等权平均；其余未来位置保留删失或未观察掩码。概率预测仅为后续独立P分支，**本轮不做MDN、Diffusion或概率损失**。

本次实验只做N0（冻结数据对接与评估合同）、N1（五种独立Backbone的公平比较）、N2（固定N1在每折内部验证集选择的Encoder，对比五种Decoder）。**不能沿用原SR-111模型组件**；旧M1作为独立历史强Baseline，不能用其模型权重初始化新模型。模型使用简洁前缀VIQ和ΔQ(V)；全局/局部显式物理统计特征的引入属于后续单独控制实验，不要这次偷偷加入。

## 1. 已提供工程代码

收到交付包 `N012_Codex_handoff.zip` 后，将 `n012/`、`configs/N012_config.json`、`tests/test_n012.py` 解压到 `battery` **根目录**。附带 `README_ZH.md`、`STATUS.json` 供审计参考。现有独立代码包含：

- `n012/data.py`：MATR/HUST资格守卫、真实后缀掩码、cell-disjoint folds、train-only scaler；Stanford按当前缺失的原始循环认证fail-closed。
- `n012/model.py`：五种Encoder（`resmlp`, `voltage_cnn`, `temporal_tcn`, `cnn_tcn`, `cnn_transformer`）及五种Decoder（`query_mlp`, `legacy_flow`, `free_rate`, `monotone_spline`, `physics_basis`）；绝不调用`models/unified_matr/`的SR-111算子。
- `n012/run.py`：逐折训练、内部val选择、真实test预测、按电芯与时距评分、独立保存权重与完整原始预测。
- `tests/test_n012.py`：7项工程测试。此前仅合成数据烟测，不存在真实数据N1/N2评分。

**先审计这些代码再跑正式实验**：验证是否存在测试信息泄漏、数据集身份混淆、失效掩码未过滤、SOH参考定义不一致、输出参数错误、冻结数据版本变化、性能瓶颈、checkpoint原子性与恢复逻辑缺失。允许在新目录内修复并写下变更依据；不要以测试结果为依据改变任务/分折/特征定义。测试指标读取必须在当前折的候选选择和超参数冻结之后进行；如原脚本在选模前打印候选test MAE，应调整为冻结后统一输出，避免人为test驱动选模。

## 2. N0 数据合同（先执行，PASS后才训练）

### MATR：正式主队列

- 调用现有 `scripts.unified_matr.train_seed42.load_frozen_data()`，核验 `data/unified_matr_seed42/DATA_FREEZE.json`、原始循环认证、真实source SHA、`cells.parquet`、`early_viq_qv_64.npz`、`future_soh.parquet`、`split_contract.json`；仓库已登记169 unique physical cells，但你必须以实际加载结果为准。
- 使用原始物理cycle 1–100，原始UID，五折旧split合同；跨批续测同一电芯不可跨拆分。学习输入不能看到任何未来SOH或未来工况。
- 统一输出SOH的名义容量归一化口径，保留各电芯完整已观察后缀和未观察mask；不可只挑能到1000圈的长寿命样本。

### HUST：第二个正式单域任务

- 读取 `data/hust_raw_idea_v2/DATA_QUALIFICATION.json` 的 `DATA_QUALIFICATION_PASS`，采用 `views/actual2C_loaded_response/early_inputs.npz`（非稀疏5C充电视图），`future_soh.parquet`和真实循环键；仓库已审计77颗，但仍以实际资格为准。
- 保留其约2C的D3放电窗口（2.8–3.1 V）、容量定义、单独的fold-local标准化和已知策略字段；不得将MATR 4C的绝对Q(V)物理电压bin和HUST同索引bin解释为同一物理量。
- 固定5折cell-disjoint划分并保存UID清单及SHA；每种精确协议一般缺少重复，不能宣称测试了组内cell variance。

### Stanford：单列数据状态，不做虚假同任务实验

- 当前仓库审计181颗供应曲线，但其 `cycles_interpolated` 经BEEP插值和重编号，缺乏与MATR严格一致的**真实原生cycle 1–100**完整训练张量。优先搜寻本地是否已存在后续恢复的原生认证文件、完整记录及合法前缀/后缀；若已有证据，先完成单独的正式数据资格、时间坐标和SOH合同，再加入N1/N2。
- 若仍未满足，**明确记录Stanford `N0_BLOCKED`**，不把供应K100伪装为真实K100。可另起名为 `SUPPLIED_CYCLE_EXPLORATORY` 的独立任务，但不能与主实验数值混合或声称同定义。

N0必须输出：资格表、原始/供应循环语义、每域N/后缀观测数/删失数、前100圈有效曲线支持、train/val/test独立UID、每fold输入标准化来源、源文件hash、失败原因、未来输入泄漏QA。**若N0失败，不允许悄悄降低门槛继续训练。**

## 3. N1 五类Backbone公平比较

任务信息预算：`[B,100,64,4]` 的 `V/I/Q/ΔQ(V)`，有效点mask、原始cycle IDs、预测前已知policy及SOH100前缀锚点；输出为未来query cycles上的完整SOH。

固定 `query_mlp` Decoder，比较：

1. ResMLP
2. VoltageCNN
3. TemporalTCN
4. VoltageCNN + TCN
5. VoltageCNN + TinyTransformer

建议初始宽度64，AdamW，batch8，150 epochs，lr=3e-4，weight_decay=0.01，seed42；必须统一配置与训练预算，在每域每个外折单独执行5个候选。准确报告参与预测的参数量、完整训练时间、train/validation曲线、稳定性和数值异常。

**严格选模**：每个外层fold内部，使用该fold自己的validation cell-mean MAE选择Encoder；不能跨外折混合val选择一个全局Encoder，因为另一折验证电芯可能是当前折test。新颖性在于受控表征对照，不根据最优test指标倒推选型。N1结束后冻结每fold选中的Encoder及其SHA。优先让每fold选模流程闭环再统一test评分。

## 4. N2 五类Decoder受控比较

对每个外折，固定N1经validation选择的Encoder**结构**，从头初始化/训练，依次比较：

1. `query_mlp`：直接未来循环条件预测
2. `legacy_flow`：数学风格独立重写的解析状态演化，不是SR-111参数重用
3. `free_rate`：softplus非负速率自由积分
4. `monotone_spline`：非负基函数系数的单调累计衰退
5. `physics_basis`：自限制对数项 + 线性项 + 平滑加速项的低维动力学函数

上述5组只允许Decoder变化，输入、训练期和模型选择规则都固定；参数差异需要如实报出，并考虑在开发阶段补充规模匹配/相同损失的对照。对`free_rate`和结构化Decoder做导数符号、远期数值稳定、任意query密度及同一查询一致性测试。真实观测SOH允许局部回升，不能把潜在单调趋势与真实测量轨迹混为一谈。

**核心科研判据**：如果`physics_basis`只优于`free_rate`但不优于`monotone_spline`，只能声称形状约束有益，不能宣称三分量物理机制被验证。若长期时距改进主要由少数电芯主导，必须披露反例而非报告单一平均数。

## 5. 训练实施、严谨评估与防止伪完成

- 不要在缺失真实MATR/HUST数据时使用SYNTHETIC替代正式训练。合成数据只允许工程smoke和单测。
- 固定外折/val，不以test反复调学习率、结构或类别；恢复训练必须含optimizer、RNG和选权状态，重启不得更改epoch和样本采样。
- 优先完成一次真实数据单折单模型smoke（前向/反向、有限梯度、正确val/test掩码），再一次性跑完整注册矩阵。若CPU或GPU环境限制，报告真实完成单元和剩余单元；禁止编造MAE或空表。
- 每颗电芯完整可观察后缀等权，分时距报告101–200、201–500、501–1000、1001+的MAE/RMSE、有效电芯数及观测点数。另单独报告各域EOL80资格（HUST 0真实事件）和协议分组指标。旧M1在MATR的0.018762只能作同口径历史参照，不作为新独立盲测。
- 完成至少3个种子的最终关键对照以估计训练波动（若算力允许，所有矩阵重复3 seed；不足时优先seed42完整N1/N2，再为最终候选和两个强Baseline扩充种子；必须明确完成度）。配对bootstrap按电芯及协议组，且承认不能替代跨种子方差估计。
- 同时保存原始逐电芯预测数组、真实标签、mask、测试集UID、每折内val选择日志、完整训练日志、参数量和数据指纹；不能只给一个摘要图。任何无效结果保留负例，不根据分数删除。

## 6. 具体命令（仓库根目录，CUDA已可用时）

```bash
python -m pytest -q tests/test_n012.py
python -m n012.run --repo-root . --domain MATR --phase N0 --out results/N012_independent
python -m n012.run --repo-root . --domain HUST --phase N0 --out results/N012_independent
python -m n012.run --repo-root . --domain MATR --phase N1 --epochs 150 --device cuda --out results/N012_independent
python -m n012.run --repo-root . --domain HUST --phase N1 --epochs 150 --device cuda --out results/N012_independent
python -m n012.run --repo-root . --domain MATR --phase N2 --epochs 150 --device cuda --out results/N012_independent
python -m n012.run --repo-root . --domain HUST --phase N2 --epochs 150 --device cuda --out results/N012_independent
```

先审计脚本与GPU资源，必要时修复真正的工程问题，重跑单测。**不要仅复制上述命令却没有实际执行。** 完整训练过程以程序日志、checkpoint及各单元真实梯度更新计数为凭证。若使用AutoDL，确保当前实例有数据卷，且设置了合适的CPU数据加载与线程数。

## 7. 必须交付

1. `N0_DATA_AUDIT.md`：MATR/HUST/Stanford资格、输入输出定义、数据指纹、split UID、防泄漏检查和阻断原因。
2. `N1_RESULTS.md`：五种Encoder在MATR/HUST逐折与总体成绩、训练配置、训练耗时、参数量及验证选型结果。
3. `N2_RESULTS.md`：五种Decoder在固定Encoder条件下逐折与总体成绩、时距诊断、反例、相对M1历史结果的同定义比较。
4. `N012_FINAL_RESEARCH_ASSESSMENT.md`：哪个Insight被支持、哪个被证伪、是否存在种子方差/协议O O D问题、下一步是否值得做N3或概率支线。注意不能用结论包装失败。
5. 对应源码、配置、测试、checkpoint、训练曲线、完整per-cell预测、结构/数据/分割SHA和真实执行验收清单；保存新目录独立于旧实验，不覆盖旧权重。

**不得把“生成了代码”“通过了合成数据测试”写成“真实实验完成”。** 如无法访问仓库或GPU，精确说明被阻断的N0/N1/N2步骤，交付可接续的代码和检查结果，不产生虚构数值。做完所有实际可执行部分再汇总报告，不以询问用户作为默认停机方式。
