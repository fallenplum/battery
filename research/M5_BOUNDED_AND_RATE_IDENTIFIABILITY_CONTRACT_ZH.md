# SRSR-Physics M5 有界动力学 + 早期 Q(V) 可辨识性：统一固定执行合同

研究日期：2026-11-16。父版本：M4 消融冻结提交 `565a638a52093891842b7278fec2c99d28258ccd`。本文件同时授权两条互不回写的实验流，但正式模型**只有一臂 M5**，不再发散更多结构。

## A. 唯一科学问题

**能否从 MATR LFP/石墨电池真实循环 1–100 的共同 4C 放电 Q(V) 响应及已知快充策略中，推断其未来平均退化速率水平及阶段变化，并用有界低维动力学外推完整 SOH？**

- 任务、数据和划分原样复用 `data/unified_matr_seed42/` 的 169 电芯认证物理轴及 frozen SHA。电压 2.8–3.3V、64 点、20 global、64 local residual、8×26 local、前缀 100、SOH=原始放电容量/1.1Ah，未见协议固定46电芯。不得以实验得失增减电池、更换未来监督定义或重采样测试集。
- 注意：**当前没有真实认证的 Knee-onset 事件标签**。任何 Hinge 候选、TDA 阶段边界或可学习 transition center 均不得写成物理 Knee 真值。
- 旧有 M1、A1、M4 全部历史测试均已暴露；本轮属于相同数据的内部回顾性验证，严禁使用“新盲测”“因果证明”。

## B. M5：仅替换动力学解码器，必须先训练完整模型

固定编码器为 M4 的 global/local response + dynamic persistence + 两层真实 B 写入；固定全部训练设置 seed42、150epochs、batch8、AdamW lr3e-4、权重衰减0.01、val first-min、完整未来轨迹监督与原两个区间速率辅助 loss。**不修改数据、物理响应特征、持续性、优化器或 loss 权重。**

新模型源码：`models/unified_matr/m5_bounded.py`，直接训练入口 `scripts/unified_matr/train_m5.py`。M5 相对于 M4 只替换 rate flow 为非负、上界为2.0/1000真实循环的**相对容量损失强度**，并用指数映射生成潜在 SOH：
\[
\tau=(k-100)/1000,\quad
\lambda_i(\tau)=2\,\mathrm{sigmoid}\left(
\mathrm{logit}(0.05)+a_i+b_i\tanh(\tau/2)+c_i\tanh^2(\tau/2)+d_i G_i(\tau)\right),
\]
\[
G_i(\tau)=\sigma((\tau-\kappa_i)/w_i)-\sigma(-\kappa_i/w_i),\quad
SOH_{i,\mathrm{trend}}(k)=SOH_i(100)\exp\left(-\int_0^\tau\lambda_i(u)\,du\right).
\]
参数 a/b/c/d 由有界 tanh 输出预测；kappa 的模型允许范围为(0.1,3.6)千循环相对 100 的距离，w∈(0.05,0.85)。`kappa`只是隐含软转变中心，不是 Knee 监督或保证可辨识的物理事件。固定 `lambda_max=2`、初始 `lambda≈0.1/1000 cycles` 是训练前注册的工程参数，不允许基于测试集微调。

逻辑优势：100圈锚点精确、非负潜在趋势、非增且积分不产生负 SOH、有限强度、长程有界；不足：物理机制的真实单调性与曲率未唯一识别，模型参数可能相互替代，过强上界可能欠拟合真实快速衰退。保留原 M4 0.03 观测残差（观察序列可小幅上升），未来只在真实测得循环上计算 MAE 与 RMSE。对超出随访至5000圈的输出单列未验证数值稳定诊断。

从仓库根目录的原训练机器直接执行：
```bash
python scripts/unified_matr/train_m5.py --data data/unified_matr_seed42 --output-root results/unified_matr_m5_bounded_seed42 --methodology-contract research/M5_BOUNDED_AND_RATE_IDENTIFIABILITY_CONTRACT_ZH.md --mode register --device cuda
python scripts/unified_matr/train_m5.py --data data/unified_matr_seed42 --output-root results/unified_matr_m5_bounded_seed42 --methodology-contract research/M5_BOUNDED_AND_RATE_IDENTIFIABILITY_CONTRACT_ZH.md --mode smoke --device cuda
python scripts/unified_matr/train_m5.py --data data/unified_matr_seed42 --output-root results/unified_matr_m5_bounded_seed42 --methodology-contract research/M5_BOUNDED_AND_RATE_IDENTIFIABILITY_CONTRACT_ZH.md --mode train --device cuda
```
原 M4 的训练器会负责数据哈希、权重选取、记录、断点恢复、六单元测试，M5 在运行时只替换模型工厂及注册元信息。总计 6×150 轮、12,450次正式更新；若 smoke 的梯度/确定性/掩码未通过，**禁止开始正式训练**。结果保存 M5 自己的 `results/unified_matr_m5_bounded_seed42/`，不覆盖 M4、M1、A1 历史产物。正式测试成绩产生后与保存的 M1(0.018762)、A1(0.020148)、M4(0.023271) 五折和未见协议值对比，不能根据哪个更优选择模型定义。

## C. 可辨识性：不使用 M5 测试误差决定标签或架构

分析入口：`scripts/unified_matr/analyze_m5_identifiability.py`，复用 `load_frozen_data`、169个前缀、原五折 train/validation/test；训练集内拟合缺失值与标准化，验证集选择 Ridge alpha（固定 [.01,.1,1,10,100]），只在该折测试电芯评分。前缀输入 C0 = 已知policy、policy mask、SOH100锚点；C1 = C0 + 最近一个合格前缀循环的全局20 Q(V)特征、8个局部电压段的signed mean/RMS及掩码；全部输入来自原始循环≤100。

四项预先固定的监督：
1. `level_mid_all`: r[200,500]，n=151；
2. `level_mid_shared`: r[200,500]，但只在有两个阶段标签的同42颗；
3. `level_late_shared`: r[500,1000]，同42颗；
4. `change_late_minus_mid_shared`: r[500,1000]−r[200,500]，同42颗。

评价每个任务 C0/C1 的真实周期单位 MAE、同一电芯配对偏差及对训练均值的标准化 skill、预测与观测 Spearman、预登记 protocol 分组指标。所有42颗具有两个未来窗口的样本是**寿命足以被观察到1000圈的选择性群体**，其 sign(Δr) 本身不能当成预测能力证据，重点检验 Δr 的大小。此前由封存原始五折标签重算得到：中期标签151颗，远期42颗、两者共有42颗；共有集合的42颗均晚期平均速率较大，速率比中位数4.7865、跨阶段Pearson0.5748。该统计不使用早期 Q(V) 模型，不能当成可辨识性已得到验证。

```bash
python scripts/unified_matr/analyze_m5_identifiability.py --data data/unified_matr_seed42 --output results/unified_matr_m5_identifiability_seed42
```

**关于 Knee：不应把“不能精确辨识瞬时转折点”当成已证实命题。** 本轮只能报告拟合型 Hinge 候选 `kappa_proxy`：改变未来拟合起点（101、201、301真实循环）、对同自由度二次平滑函数比较拟合、统计5% SSE剖面宽度，判定从轨迹上定义的候选坐标有多依赖模型选择。该算法使用未来观测仅生成**非监督诊断结果**，不会被输入 M5 或早期 Q(V) 速率回归。Hinge 可以在光滑指数/二次曲线上产生“稳定拐点”（原 D0 负对照已证实），所以不能用它计算物理 Knee 的精准率。M5 另保存 soft_transition_center_cycle/soft_transition_width_cycles/soft_transition_signed_amplitude 三项诊断，用于检查同等轨迹精度下的参数稳定性；参数是否可辨识仍需多种子或剖面分析。只有后续有独立真实 Knee 认证，才能真正验证“早期 Q(V) 无法精确估计其时机”。

## D. 交付与科学门槛

两条线完全并行但**选择互不依赖**：
- M5 输出六折 `MATRIX_COMPLETE.json`、`COMPLETE.json`、`SELECTED.pt`、训练日志、测试逐电芯SOH、速率、EOL与机制诊断；整理完整 SOH MAE/RMSE、4时距分层、未见协议组宏平均、MATR_b1c2 已知反例及随访外clipping。
- 可辨识性输出 `rate_predictability_oof.csv`、`rate_predictability_summary.csv`、`knee_proxy_stability.csv`、`IDENTIFIABILITY_CONTRACT.json`。说明每个输出目标样本分母、协议组、单seed历史暴露及候选 Knee 非真实事件。
- 如果 C1 改善速率水平但不改善 Δr，只能说现有早期 Q(V) 特征/模型对变化量的证据不足，**不能推断变化量物理上不可辨识**；如果 Hinge 候选不稳，只能说明候选标注不可靠，**不能断言真 Knee 不存在**。
- 如完整 M5 失败，保留负结果；不得返回修改数据、改变窗、挑掉长寿命电芯或不断搜索解码头。

## E. 当前执行事实

在本次对话环境，`m5_bounded.py` 的独立相对强度解码头已通过 CPU 初始锚点/非增/有界/梯度/乱序查询 smoke；两份正式模型文件和分析脚本已推送新 GitHub 分支。但云端 Codex 工作区调用返回内部故障，当前本地无 GPU 和完整 MATR 冻结二进制张量，因此**尚未执行 M5 六折真实训练，亦尚未执行基于冻结 VIQ 的可辨识性回归**。不允许把脚本已提交报告为模型已跑完。
