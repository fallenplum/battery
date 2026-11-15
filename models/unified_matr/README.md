# MATR 统一任务七组模型工程

## 当前状态

本目录提供从头初始化的 seed42 七组模型入口及工程检查。工程封存时正式模型 optimizer updates 为 0。DATA-FREEZE 已通过580项核验并绑定44个文件，固定169颗输入合格电芯、123431个未来有效观测点、两段rate支持151/42，nominal0.8 EOL事件169颗。CPU 与 CUDA 工程检查使用人工张量验证实现性质，模型效果由后续唯一seed42、42单元正式训练产生。

## 数据接口

```python
from models.unified_matr import UnifiedMATRModel, model_registry

model = UnifiedMATRModel(
    arm="M3",
    policy_dim=policy_dimension,
    global_dim=global_dimension,
    local_feature_dim=local_dimension,
    seed=42,
)
prediction = model(**batch)
```

| 参数 | 形状与约束 |
|---|---|
| `raw_viq` | `[B,100,64,3]`，V、I、absolute Q 的 fold-train 标准化视图 |
| `point_mask` | `[B,100,64]`，布尔，记录实际合格放电支持 |
| `cycle_ids` | `[B,100]`，有效项为严格递增的原始整数循环号 1–100 |
| `reference_viq` | 与 raw 相同形状，旧 SR111 的早期参考容量尺度归一化视图 |
| `policy, policy_mask` | `[B,P]`，预测前已知协议数值及布尔可用掩码 |
| `global_features, global_mask` | `[B,100,G]`，公共偏移、整体形变及其全局动态，禁止夹带局部残差统计 |
| `local_residual, local_mask` | `[B,100,64]`，保留电压位置的 residual 及布尔掩码 |
| `local_features, local_feature_mask` | `[B,100,8,L]`，八个固定电压窗口的局部统计及布尔掩码 |
| `anchor_soh100` | `[B]`，同口径早期可观测 SOH100 趋势锚点，所有组共用 |
| `query_cycles` | `[T]` 或 `[B,T]`，已知查询坐标，至少 100，真实未来标签与监督掩码留在模型外部 |

返回 `trajectory`、`trend`、`observation_residual` 和标量 `persistence_penalty`。速率参数化组另返回 `rate_per_1000_cycles`。`return_context=True` 可取得末端 state、condition、record_valid、raw_trend 和查询坐标。

`raw_viq` 与 `reference_viq` 使用 fold-train 数据拟合各自尺度。参考视图中 `N_V=R_V`。每颗电池的参考尺度为合法早期合格放电循环内 Q 的 95%–5% 分位跨度的中位数，I 与绝对 Q 分别除以该尺度，再使用当前 fold 的 train 参考统计标准化。保留原 Q 偏置。分解所需 `Q_ref(V)` 单独构造。

## 模型与信息消融

| 组 | 输入与机制 |
|---|---|
| M0 | 原 VIQ 与参考视图、已知策略，原 SR111 局部选择器、两层 history Mamba、原解析 Flow |
| M1 | M0 加 global 与 local 的普通逐圈融合 |
| M2 | 与 M1 相同观测，逐圈物理内容及真实循环间隔控制的慢变选择器条件 |
| M3 | M2 编码器加非负容量损失率与积分趋势 |
| M3-NL | global、policy、anchor 与支持掩码；完全删除 raw/reference 数值及全部 local 数值和 local 掩码 |
| M3-NP | 与 M3 相同信息，恢复逐圈普通融合 |
| M3-NR | 与 M3 相同编码器及共同观测层，恢复原解析 Flow |

M3-NL 使用有效 global bottleneck 与八个 global tokens 接入原局部算子。该组是全局信息消融。所有 raw/reference 曲线及显式局部输入改变为 NaN 后，输出仍逐值一致。它同时移除原始波形中的局部信息与电流数值信息，因此解释结果时必须保留这项范围说明。global 动态的列级来源需由数据冻结清单另外认证。

M2 与 M3-NR 按合同得到相同计算图和相同初始化。它们保留为合同的重复一致性核对组。

所有组共享有界观测层，定义 `0.03 * tanh(g(state,condition,tau))`，`tau=(cycle-100)/1000`。趋势精确锚定 SOH100，观测层允许早期短时回升与 anchor 测量偏差。观测 SOH 输出统一截取至 `[0,1.2]`。M0 因而属于旧 SRSR 与旧解析 Flow 在单放电、已知策略及共同观测层下的任务适配。原解析式 `exp(-rate*t)*state + (1-exp(-rate*t))*mean` 与 decoder 差值保持原实现，native SOH correction 使用旧编码换算系数 0.2。

M3 系列趋势为共享 SOH100 anchor 减去 softplus 正速率的逐循环梯形积分。观测残差可表达局部波动。慢变分量采用可学习逐通道保持系数和真实循环间隔，允许正、负漂移。建议训练端固定 `1e-3 * persistence_penalty`，仅用于 M2、M3、M3-NL、M3-NR。M1 与 M3-NP 的该 penalty 精确为 0。

持续性条件采用dense causal EMA展开，参数为原64个retention logits。独立 `persistence_oracle.py` 保存原loop公式，28组CPU/CUDA、FP64/FP32比较覆盖state、penalty及value/cycle/retention梯度，包括首尾与内部缺失、allmissing、单有效点、非均匀/倒序gapclamp1及饱和retain0/1。FP32 state最大绝对差4.77e-7，最大gradient差5.69e-9。预先声明的FP64/FP32容差全部通过。

## 参数预算与初始化

解析 Flow head 与速率 Flow head 均有 2,752 个实际参与预测的可训练参数。M1、M2、M3、M3-NP、M3-NR 参数数完全一致。M3-NL 用实际 global bottleneck 宽度近似匹配，报告实际差值。M0 自然少于新增物理编码器的各组。

已登记实际接口维数 `P=15,G=20,L=26`。M0 为 101,901，五个完整信息组均为 134,301，M3-NL 为 134,331，差值 +30。输入含十四个旧策略数值列及实验前已知的 30℃ 箱体设计设定。正式执行还会根据冻结数据逐项登记 `model_registry`。所有组的共同 state keys 在形状相同时逐值相同。每个已注册参数在有支持的工程张量上均取得非零有限梯度。

## 来源与工程核验

旧源码身份、SHA256、数学复用范围和新接口适配记录在 `SOURCE_LINEAGE.json`。CPU 的 95 项、CUDA 的 102 项工程检查均通过，使用上述实际接口维数。回执分别为 `ENGINEERING_SELFCHECK_CPU.json`、`ENGINEERING_SELFCHECK_CUDA.json`。检查覆盖原 SSD、原 history Mamba、原 SR111 选择器和解析 Flow 的数值一致性，共同初始化逐值一致性，mask 隔离，未来输入拒绝，NL 删除，速率非负，趋势单调，七组相同的有界观测公式，参数梯度可达性及 CPU/CUDA 预测一致性。

两份模型 QA 均启用正式入口相同的 strict deterministic algorithms 与 CUBLAS4096:8，查询覆盖 dense cycle100–5000 及离网格 200.5，七组均完成 forward/backward。原先 CUDA QA 缺少 strictdet 配置覆盖，后由正式入口诊断发现 CUDA cumsum backend 错误，已在科学训练前修复。`deterministic_cumsum` 在 CUDA strictdet 模式下将小型 prefix/rate tensor 转到 CPU 累加后返回原 device，保留 dtype、自动微分与数学公式，其余执行沿既有 cumsum。数据复制的性能开销需要正式运行记录。

训练入口为 `scripts/unified_matr/train_seed42.py`。独立真实仪器循环映射 QA、四份来源档案、所有消费数据与配置 SHA 共同通过后才能进入训练。runner 固定七组、seed42、150 epochs、batch8，并核对实际 optimizer updates。断点原子保存模型、optimizer、所有 RNG、选中权重与 history，恢复时依据 checkpoint 重建 journal。测试前冻结 selection，完整完成后只核验已有文件。矩阵使用既有 OS-held execution lock 保证单写入者。`ENGINEERING_RUNNER_SELFCHECK.json` 记录 24 项无拟合检查。

独立 `scripts/unified_matr/splits.py::build_splits(cells_rows, configsplit)` 只读取 `cell_uid` 与 `protocol_uid`，先给全部固定电芯分配角色。五折外 test 在每个 exact protocol 内及全队列人数差均至多一颗，validation 按各 protocol 的 cell 比例分配 20%。敏感性独立 hash 选择四舍五入的 20% exact protocol groups 作为 test，其余 cell 再分 train/validation。输入资格、未来标签及长度、continuation 描述改变后，角色及 fingerprint 逐值不变。`ENGINEERING_SPLITS_SELFCHECK.json` 记录 20 项无拟合检查。

## 训练后分析

`scripts/unified_matr/analyze_results.py` 在42个单元和矩阵完成标记、冻结数据、全部单元哈希通过后读取 test 表。源码已经实现分析，科学分析尚未执行。`ENGINEERING_ANALYSIS_SELFCHECK.json` 记录26项纯数学、样例张量与完整性守卫检查，formal_results_read 为 false。

主五折 OOF 与协议外推分别输出指标及固定对照 CSV。分析包含 cell-mean 轨迹 MAE/RMSE、四个 horizon、protocol-macro、两段 exact-endpoint rate、固定七个 contrast 的2000次 seed42 配对 cell/protocol bootstrap。每电芯、每协议、资格分母、全部负向与缺支持记录、训练 epoch/参数/time 共同保存。M2 与 M3-NR 对预测、rate、EOL、训练历史及选中权重 tensor 逐值核对。

训练入口额外保存 `test_rate_predictions.csv` 与 `test_eol_predictions.csv`。EOL 附属分析采用独立预设至 cycle5000 的 trend 首次0.8 crossing，未 crossing 风险分数保持5000。Harrell C 风险集限于 cycle100 后仍有可用结局随访的电芯，已观察事件时间严格早于另一事件或删失时间构成可比较对，结局 ties 排除，预测 ties 计0.5。EOL 同时报告事件、右删失、可比较对与配对 bootstrap 有效重采样数。

```powershell
& D:\Anaconda\envs\tslib\python.exe scripts\unified_matr\analyze_results.py `
  --data DATA_ROOT --run-root RUN_ROOT --output ANALYSIS_ROOT --report REPORT_PATH
```

这些 root/path 参数应对应已完成的正式冻结数据、42单元训练目录及结果交付目录。中文报告、CSV、JSON 与分析合同逐项封存，完整完成后再次执行会核验已有交付。

## 工程运行配置

`ENGINEERING_GPU_PROFILE.json` 和 `report/UNIFIED_MATR_ENGINEERING_RUNTIME_ZH.md` 记录batch8、query101–2500、FP32/strictdet、intra-op4及正式默认inter-op24的工程测量。每组每backend为2次warmup和3次forward/backward。M2/M3/NL/NR的loop到vectorized时间比4.76–5.86，GPU allocated峰值约200.50MiB、reserved216MiB。87150updates的纯forward/backward投影约0.645小时，完整训练另有优化器、验证和checkpoint成本。短测、控制组波动与同时运行的数据构建进程均有记录。工程投影置信度中。

```powershell
& D:\Anaconda\envs\tslib\python.exe -m models.unified_matr.selfcheck `
  --legacy-root D:\develop\patchtst\battery_ai\_srsr_viq_physics_response_v1\runtime `
  --output models\unified_matr\ENGINEERING_SELFCHECK_CPU.json
```

优势是沿用已存在算子并为普通融合、持续性与速率机制提供相同预算对照。风险包括单 seed 首轮缺少训练波动重复证据，M0 的任务几何与已知协议适配，以及 NL 全局信息消融同时移除多个原始曲线信息来源。正式结论还需要真实循环轴、数据冻结和完整科学运行支持。
