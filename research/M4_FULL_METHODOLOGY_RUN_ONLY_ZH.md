# SRSR-Physics 完整 Methodology 单臂正式执行令（先跑完整模型，暂不消融）

**日期：2026-11-15 | 基线提交：c81aba21d71b337e9fa5f4d16fd37c6eb483eb12 | 状态：可交由具备 MATR 冻结数据与 GPU 的 Codex 执行；本对话尚未启动真实训练。**

## 0. 唯一目标、禁止事项

用户要求“先直接跑 Methodology 看看最终效果，跑完再逐一消融”。本次只实现并运行**完整新模型** `M4_FULL_METHOD`：I1 公共—局部响应分解 + I2 响应条件化动态持续性及真实 Mamba B 写入调制 + I3 非负有效损失速率积分 + 小幅观测残差。禁止在本轮新增、训练或选择任何消融模型；只允许 engineering unit tests、synthetic smoke 和 frozen-data real train。参考旧 M0/M1/M2/M3 已保存成绩仅作历史对照，不重新训练或用其测试集挑参。

本次任务保持不变：MATR 169 颗 LFP/石墨电芯，真实老化循环 1–100 的共同放电 VIQ/Q(V) 和预测前已知快充策略 → 全部实际可观测未来 SOH。五折 cell-disjoint OOF + 原来固定的 unseen-protocol sensitivity；不更换数据、不改协议、不改已冻结特征或 SOH 口径、不做 LODO。主指标 cell-mean full observed future SOH MAE 与 RMSE。

## 1. 使用已完成的数据冻结，禁止新的“边跑边补审计”

直接读取 `data/unified_matr_seed42/DATA_FREEZE.json`、`early_viq_qv_64.npz`、`early_physics_features.parquet`、`future_soh.parquet`、`split_contract.json`；全部 SHA 校验和现有 `load_frozen_data` 守卫必须 PASS。禁止修改任何 `data/unified_matr_seed42/` 文件。当前 QA 580 项 PASS；处理了 174 份原始 CSV/仪器元数据/协议，169 unique physical cell；前缀 VIQ [B,100,64,3]；global [B,100,20]、local residual [B,100,64]、local feature [B,100,8,26]；已知 policy [B,15]；实际已观测后缀完全保留 mask。MATR 4C 共同放电窗 2.8–3.3 V，仪器 cycle IDs 已认证。

* 不把 loaded Q(V) derivative 说成 OCV/真实 LLI、LAM。
* 禁止使用未来观测、EOL、未来可变工况作为输入；训练 scaler 仅拟合本折训练电芯。
* 所有模型从随机初始化训练；不把同一份实验中曾看过的 test 结果称为全新 blind test。

## 2. 完整新模型实施细节（单模型 M4_FULL_METHOD）

**I1 Global–Local Response Decomposer：** 直接复用已冻结、带掩码的稳健投影 `dQ_n(V)=a_n+b_n Q'_ref(V)+r_n(V)`，保留 `a,b`、64 点 r(V)、8 个预设局部电压段及相应真实循环差分；保持原 `PhysicalResponseEncoder` 的全局/局部通路与位置编码。不要因为某些缺失值放宽资格、换窗口或改锚点。

**I2 Response-conditioned Selective Persistence：** 目前代码 `SoftPersistence` 只有通道级固定可学习 `rho`；这不是导师方案中的新方法。新增一个 `DynamicPersistence`：
- 每个真实有效循环的输入来自该圈已冻结的 local/global physics token、先前圈构造的稳健漂移/趋势、局部有效比例与噪声代理、已知快充 policy、真实 `Δn`（全部≤100）；
- `rho_n=clip(sigmoid(MLP(...)),0.01,0.99)`，可为按通道 [B,100,64]，在缺失圈保持状态；第一次合格圈初始化为当前有效响应，不向前借用缺失未来；
- `z_n=rho_n**Δn*z_prev+(1-rho_n**Δn)*phi_n`，记录可解释性诊断：每周期 rho、写入率、实际 cycle gap、mask、漂移方向一致性；
- 由 `z_n` 与观测质量生成 `g_n=sigmoid(W_g(...))`，**真实调制跨循环 Mamba2 的 B_t（history mixer）**：在 `MaskedMamba2` 对 x,b,c 的卷积输出之后、调用 `dense_ssd` 之前，按当前周期输入 `b'_n=b_n + g_n * tanh(W_B z_n)`（状态维度匹配16）；仅有效周期调制，跳过缺失。对 `W_B` 采用零初始化，使新分支从原 SSM 行为附近平稳开始；记录 b 与 b' 的 L2 变化，确保确实在计算图上影响状态写入；
- 不允许只把持续性 token 拼接到普通 hidden 然后称作 B routing。严禁对 invalid masks 的梯度产生未来信息。

**I3 Rate-conditioned Degradation Flow：** 以编码状态 h [B,16]、condition [B,16]、实际未来循环查询 t>100 计算非负有效损失速率：
```
rate_i(t) = softplus(f_theta(h_i,condition_i,(t-100)/1000))
trend_i(t) = SOH_i(100) - integral_100^t rate_i(s) ds
sohhat_i(t) = clamp(trend_i(t) + 0.03*tanh(epsilon_theta(h_i,condition_i,t)), 0, 1.2)
```
速率可随时间增加、降低或保持，不强制全局加速；数值积分使用真实 cycle 查询轴和稳定的固定梯形规则。对较远 t 的数值稳定性、rate 极值、跨查询前后一致性、单调趋势和从 t=100 锚点的一致性作 smoke/QA。旧 M3 的正速率积分曾比旧 Flow 明显差，因此本轮是**面向已知缺陷的新版本**，不能声称先前 M3 已经验证该结构。遇到严重数值/梯度错误应修复原因，不允许看测试集调节参数。

**模型其他部分：** 原 `SingleDischargeFrontend`、self-reference 局部 Mamba、两层历史 Mamba、policy conditioner、state [B,16]/condition [B,16]、观测层保持，新增模块代码在 `models/unified_matr/` 下新增文件，并复用现有源码，不覆盖先前七臂权重。新的唯一 arm 名用 `M4_FULL_METHOD`；新增唯一训练入口 `scripts/unified_matr/train_methodology_full.py`，避免把之前的固定 `train_seed42.py` 七臂脚本改掉。

## 3. 只训练完整模型，一次性固定超参/输出

复用冻结的 `configs/unified_matr_seed42.json` 及 `configs/unified_matr_seed42_splits.json` 的训练/评估原则：seed42，outer 5 cell folds + 原 unseen-protocol sensitivity，150 epochs/单元，batch8，AdamW lr=3e-4, weight_decay=0.01，固定模型验证集 cell-mean trajectory MAE 的 first-min 选权，不早停，所有有效观测后缀监督；rate auxiliary weight0.1，scaled 1000，soft-persistence penalty weight0.001 除非新门控对 penalty 的量纲与旧版不兼容（若不兼容，提前明确注册并在训练数据上选择，绝不看 test）；无下一轮模型/数据搜索。

一次只训练 `M4_FULL_METHOD` 六个单元。先做1–2个 batch和1 epoch工程 smoke（不是模型性能选择），通过后运行全部150 epochs×6。不得根据前几个 fold 的测试误差改变结构；如发生OOM/NaN必须停机并报告，不回退为旧 M3 冒充新模型。

运行输出统一新目录 `results/unified_matr_methodology_full_seed42/`，不得覆盖 `results/unified_matr_seed42/`：
- `MODEL_CONTRACT.json`（计算图、参数量、可训练参数、源码 SHA、冻结数据 SHA、split SHA、训练设定）
- `SMOKE_QA.json`（shape/mask/gradient/b modulation / first-valid / temporal no-future / trend monotonic）
- 六套 train log、checkpoint、val first-min 选权冻结凭证、test 预测、逐电芯 SOH 轨迹及 rate 预测、EOL80；
- `FULL_MODEL_FINAL_RESULTS_ZH.md`：六单元完成数、loss、时距分母与四区间 MAE、完整轨迹 cell-mean MAE/RMSE，协议宏平均与未见协议敏感性、EOL C；与旧 `M1=0.018762` OOF、`M2=0.018946` OOF、`M3=0.023268` OOF 的**历史、非新盲测**数值对照；
- `RUN_VERIFICATION.json`：检查全部相同真实输入/标签指纹、selected checkpoint 正常入口重放、结果算术独立复核；只要求完整模型，无消融/七臂矩阵；
- 失败及异常全部保留；首批 forward/backward 必须可重复，尽可能检查两次完整训练一致性（若未检查要如实记载）。不能将 seed42 的配对 bootstrap 解释为跨随机种子置信区间。

**实验目的：** 在固定研究问题、输入、协议下先确定新完整 Methodology 的最终预测能力、远期失效模式和正确性，不展开 I1/I2/I3 消融。只有完整模型真实完成评分后才进入逐一消融阶段。

## 4. 原始结果的必须披露事项

旧完整 `M3` 已训练并获得五折 OOF MAE 0.023268、unseen-protocol MAE 0.046896，弱于旧物理普通融合 `M1` 0.018762/0.032188。**旧 M3 不是当前新 Methodology**，因为其持续性模块不由响应条件化控制 B 参数；不能直接复用旧 M3 的结果称“方法已跑完”。

目前会话只有 GitHub 文本读取/提交能力，本地 Python CPU 环境未挂载被冻结的 ~47MB early VIQ 和完整 parquet，也没有 GPU；所连 Codex 工作区 API 返回内部失败。因此“合同已写入”不等于“六次新 GPU 训练已完成”。具备冻结仓库数据与 GPU 的 Codex 应遵守本合同立即完成新完整模型六单元训练，不等待其他数据审计，也不另行发散研究问题。
