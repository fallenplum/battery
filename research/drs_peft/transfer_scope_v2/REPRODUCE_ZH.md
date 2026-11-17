# 总 K 校准与迁移范围 V2 的运行与复核

从 battery 仓库根目录执行。本轮使用分支 `codex/transfer-scope-calibration-v2`，父提交为 `49fd7c210b60297bcf38daaccd7535ebabf32d35`。新增正式计算在本机 RTX 4060 上运行，源模型、OUT16 字典、旧目标状态和已评分锚点沿用父实验的登记来源。协议以 [PROTOCOL_ZH.md](PROTOCOL_ZH.md) 和 [CONFIG.json](../../../configs/drs_peft/transfer_scope_v2/CONFIG.json) 为准。

## 本地输入

| 输入 | 默认位置 | 核验依据 |
| --- | --- | --- |
| MATR 与 HUST 冻结数据 | `runtime/drs_peft/data_v1/{MATR,HUST}` | 现有数据分割锁、缓存 COMPLETE 清单及逐文件 SHA256 |
| 已训练的源参考与 OUT16 | `runtime/drs_peft/onesided_fewshot_v1/references` | 源折与全部 115 颗源电芯的实际拟合 UID、检查点、Bridge/Head 和字典身份 |
| 历史 Jacobian | `runtime/drs_peft/onesided_fewshot_v1/engine` | 同源参考、字典、前缀及原循环网格的逐条指纹 |
| 已有两类 LoRA 的单位缩放快照、固定 Query 预测 | `runtime/drs_peft/{onesided_fewshot_v1,specialization_comparison_v1}` | 原锁定学习率、步数、物理 Support/Query、状态和预测 SHA256 |
| 官方 Moirai-small | 父实验登记的本地 `.cache_drs_peft/moirai_small` | 官方主干权重身份与源接口重放 |

公开结果提供复核所需的聚合、逐电芯误差、配置、回执及哈希。完整运行需要上表的本地输入，其中缓存数组、模型权重和逐点预测保存在本地运行目录。安装环境沿用本机 `tslib`。PyTorch CUDA 前向用于编码和非线性预测，CPU float64 用于正则小矩阵求解。

## 运行

下列入口用于本轮正式计算。幂等阶段根据已经冻结的文件身份继续执行。冷启动可通过命令行给出独立的 `--public-root`、`--private-root` 和 `--report-root`，供新运行保存可复核的完整产物。

```powershell
$batteryPython = 'D:\Anaconda\envs\tslib\python.exe'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& $batteryPython -m scripts.drs_peft.summarize_transfer_scope --audit-only
& $batteryPython -m scripts.drs_peft.run_transfer_scope --phase run
```

增量复核可再次运行以下汇总。它使用已评分指标和对应回执。

```powershell
& $batteryPython -m scripts.drs_peft.summarize_transfer_scope
```

## 实验路径

1. 源内选择使用三个源折、三个模型种子和五次登记 draw。四个几何方法先用线性化预测筛选，再对每个方法与 K 的前三个非零候选及 ZERO 进行真实非线性源内评分。最终依据完整观测轨迹的电芯均值 MAE 选择。LoRA 保持原有学习率及优化步数，源内搜索整体缩放 η。η=0 对应原参考预测。
2. 每折的表示标准化和 Gram 正则尺度使用该折拟合电芯，最终源版本使用原 115 颗源电芯。G 方法在每个 episode 共享校准系数，L 方法依据单个 Query 的前 100 圈输入及其与同一批 Support 的距离确定权重。联合方法对真实 OUT 非线性预测相加原参考上的 CAL2 校准项。
3. 锁定源选择、总 K 及全部 Query 身份后生成目标预测。几何 360 个预测槽与 LoRA 收缩 192 个预测槽均通过统一的身份与文件完整性校验后，评分才读取固定 Query 的真实观测行。评分窗口覆盖循环 101–5000，仅计真实观测位置。HUST 含右删失记录。
4. 汇总完整比较矩阵时，加入来自父实验的 327 个评分槽。主表和逐电芯事实写入 `METRICS.csv`，逐项比较见 `CAL2_INCREMENTAL.csv`，状态与迁移诊断见 `TRANSFER_DIAGNOSTICS.csv`。复用及实际开销见 `REUSE_MANIFEST.json` 和 `EXECUTION_LEDGER.json`。各阶段的零样本起点、回执和代码哈希保留在对应结果目录。

## 优势与限制

同一物理 Support、固定 Query、共同 SPECIALIZED 起点和两种校准空间支持配对比较共享范围与内部更新价值。局部核保留绝对权重，固定源 Gram 正则使弱覆盖时的系数可以向零收缩。L 方法的坐标和求解预算按 Query 计量，运行台账记录其额外状态及前向。

联合解依赖 f0 附近的一阶近似，预测阶段的实际非线性偏差与适配效果均进入诊断。MATR 主 Query 只有 8 颗，远期分母会减小。MATR K1 的五个登记槽对应三个不同 Support 集合。固定目标队列已有历史暴露，外推到新工况或新队列的证据有限。

MATR 补集资格审计确认 30 颗输入合格，其中 22 颗在主 8 颗之外。扩展目前仅完成前缀资格核验，主矩阵使用既有 8 颗 Query。资格清单位于 `MATR_EXTENSION_IDENTITY.json`。
