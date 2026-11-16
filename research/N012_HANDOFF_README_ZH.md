# N0 + N1 + N2 独立架构实验交付（2026-11-16）

## 执行状态（先看）

- **已完成**：N0针对原仓库的真实数据结构进行只读适配与资格守卫；N1五类独立Encoder；N2五类独立Decoder；每外层折独立按内部验证集选模；训练、测试、逐电芯记录、分时距汇总与模型权重导出；7项测试；合成数据三折、两轮训练工程连通性（27个非重复单元）。
- **未完成**：MATR/HUST真实数据训练（当前会话未挂载仓库数据或GPU）；Stanford原生100圈输入尚未构建。不存在真实电池MAE/论文结论。合成数据不得用作科学结果。
- **没有覆盖或修改原仓库**任何已封存N0/旧M1/M4结果。
- **本交付仅N0/N1/N2**：不训练概率分支、不进行正式跨域共享训练（属于N3/P）。

## 使用前提

将压缩包内的 `n012/`、`configs/N012_config.json`、`tests/` 解压到 `fallenplum/battery` 仓库**根目录**。保持原 `scripts/`、`models/`、`data/` 目录原样。

环境依赖（仓库原依赖基础上）：`torch>=2`、`numpy`、`pandas`、`pyarrow`、`pytest`。建议在已配置好GPU的原科研环境运行。

先进行单元测试：

```bash
python -m pytest -q tests/test_n012.py
```

合成工程检查，**不代表科研结果**：

```bash
python -m n012.run --domain SYNTHETIC --phase all --epochs 2 --fold-limit 3 --device cpu --out results/N012_smoke
```

## N0：真实数据门槛

```bash
python -m n012.run --repo-root . --domain MATR --phase N0 --out results/N012_independent
python -m n012.run --repo-root . --domain HUST --phase N0 --out results/N012_independent
```

- **MATR**：调用 `scripts.unified_matr.train_seed42.load_frozen_data()` 进行原有 `DATA_FREEZE.json`/源SHA/原始循环QA/全量可观测后缀校验，并交叉比对重建的target数组；采用旧 `split_contract.json` 真实物理电芯互斥五折，不重新划分。
- **HUST**：要求 `data/hust_raw_idea_v2/DATA_QUALIFICATION.json` 为 `DATA_QUALIFICATION_PASS`，使用已审核的实际2C末段放电视图和真实101圈后缀，生成新的seed42五折电芯互斥划分。每个精确放电组合只有一颗，不可声称同协议内部重复验证。
- **Stanford**：原仓库只有供应坐标重编号/BEEP插值的实验资料与部分结果，没有满足原始cycle=1..100合同的冻结训练张量；当前适配器**fail-closed**并给出报错。可建立单独的供应坐标实验合同，但不应在缺乏认证时把供应K100等同物理K100。
- 输入四通道：标准化的 `[V,I,Q, Q_n(V)-Q_ref(V)]`，形状 `[N,100,64,4]`，每域独立划分，全部模型完全同信息预算。后续全局/局部物理统计特征存在原仓库中，N1/N2中刻意不加入，留待后续独立信息消融。
- **所有标准化统计仅使用相应折训练电芯**；任何无效点均保持显式mask，未来SOH绝不进入模型输入。

## N1：Encoder筛选（固定QueryMLP）

```bash
python -m n012.run --repo-root . --domain MATR --phase N1 --epochs 150 --device cuda --out results/N012_independent
python -m n012.run --repo-root . --domain HUST --phase N1 --epochs 150 --device cuda --out results/N012_independent
```

编码器5种：ResMLP、VoltageCNN、TemporalTCN、CNN-TCN、CNN-TinyTransformer。宽度64，均使用原始Cycle ID及可用性mask、同一预测头、同一优化预算。记录真实参数量以防性能仅由容量差异驱动。

**关键：每一个外层测试折单独用本折内部验证集选择Encoder**，不跨折汇总val来确定一个所谓全局优选（会造成测试折与其他折的val重叠、选择泄漏）。跨折汇总仅作描述性结果。`N1_SELECTION.json`保留逐折选择及证据。

## N2：Decoder比较（按每外层折N1验证选择固定Encoder）

```bash
python -m n012.run --repo-root . --domain MATR --phase N2 --epochs 150 --device cuda --out results/N012_independent
python -m n012.run --repo-root . --domain HUST --phase N2 --epochs 150 --device cuda --out results/N012_independent
```

解码器5种：QueryMLP、独立重写的LegacyStyleFlow、自由非负Rate积分、单调样条、低维三分量物理启发Basis。只有Decoder变动，全部**从头训练**相同的本折Encoder结构，不复用历史SR-111训练权重。

各单位保留：`COMPLETE.json`、`BEST.pt`、`SCALER_TRAIN_ONLY.json`、`HISTORY.json`、`TEST_CELLS.json`、`test_predictions.npz`。`N2_SELECTION.json`中展示每折内部选模及真正独立的nested-OOF测试平均误差。

**两类总体误差注意区分**：旧M1正式结果为固定MATR SOH比例的逐电芯平均MAE 0.018762。本新实验必须在原始169颗、相同五折和真实后缀上做新的成对比较。HUST、Stanford不能直接混入数值均值以宣布超越。

## 实验接口与解释边界

模型计算：

```text
 [N,100,64,4] masked VIQ/deltaQ
       ↓ VoltageCNN / ResMLP (周期内)
 [N,100,64]
       ↓ TCN / TinyTransformer / Identity (跨周期)
 [N,64] + known policy
       ↓ independent query-conditional trajectory decoder
 [N,T]  SOH forecast for actual query cycle IDs
```

正式模型不会调用`models/unified_matr/`下的自参考Mamba/Flow模块。`LegacyStyleFlow`是代码独立的数学对照，不是原SR-111原样重放，因此旧M1继续作为独立外部基线。

**物理基函数**：累计损失由对数自限制项、线性项、平滑激活项组成；单调样条为可替代的有界形状对照。这些结构只保证有限的数学性质，不代表已识别真实LLI/LAM/SEI机制。真实SOH可以出现观测回升；两个趋势Decoder本轮只预测单调潜在退化趋势。

**尚未覆盖**：多种seed完整训练波动、域间训练/留域评测、真实EOL80删失指标、概率校准、进一步参数量匹配与不同未来时距训练权重。这些是N3/P或者扩展审计，不应用本轮自动脚本结果声称已完成。

## 优势 / 劣势

- **优势**：SR-111结构依赖已解除；所有对照有清晰因果变化维度；旧数据封存完好；严格分折选择、可观测后缀评价和非负性数学检查；真实数据装入后同一套脚本可运行。
- **风险**：当前没有GPU和真实文件时无法验证科研效果；Stanford未认证原生轴；本次模型结构大小存在差异，需参数量对齐；N2所选择的Encoder可随外折变化，这使嵌套CV无选择泄漏，但不等于已经确定唯一的最终论文Backbone。
