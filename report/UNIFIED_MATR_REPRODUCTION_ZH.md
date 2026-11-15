# MATR seed42 结果复核与从头复现

## 数据和实验范围

统一任务使用169颗MATR LFP/石墨电芯，输入限于真实老化循环1–100的合格放电VIQ、同前缀物理响应特征与预先已知策略。主目标覆盖每颗电芯实际观测到的全部有效未来SOH。主SOH为设备放电容量除以标称1.1 Ah。历史169队列已参与探索，本次五折结果属于队列内部评估。

训练种子仅为42。七组模型各执行五个电芯互斥外折及一次未见协议敏感性分析，共42个训练单元，每个单元150轮。模型、划分、超参数和选权规则保存在冻结配置及运行注册中。

| 内容 | 路径 |
|---|---|
| 用户总合同快照 | `contracts/SRSR_Physics_统一研究问题_一次性数据处理与模型实验总合同.md` |
| 数据冻结证书及44项指纹 | `data/unified_matr_seed42/DATA_FREEZE.json` |
| 完整580项数据核验 | `data/unified_matr_seed42/dataset_qa.json` |
| 训练合同和固定划分 | `configs/unified_matr_seed42.json`、`configs/unified_matr_seed42_splits.json` |
| 42个完整训练单元与注册 | `results/unified_matr_seed42/runs` |
| 主OOF、协议敏感性及逐电芯统计 | `results/unified_matr_seed42/analysis` |
| 独立重放和统计核验 | `results/unified_matr_seed42/FINAL_SCIENTIFIC_VERIFICATION.json` |
| 全部七组及七个固定对照结果图 | `report/figures/unified_matr_seed42` |

## 环境

从仓库根目录运行命令。本轮实际环境为Windows、Python3.11、PyTorch2.4.1与CUDA。确切版本以运行和独立验证回执为准。依赖范围保存在根目录`requirements.txt`。

```powershell
python -m pip install -r requirements.txt
```

CUDA运行采用确定性算法、FP32、关闭TF32以及每个进程4个CPU计算线程。训练入口在导入Torch前登记CUBLAS确定性配置。最终权重重放应优先使用相同软件和硬件环境，跨平台浮点差异需按验证回执中的容差判断。

## 复核随附权重和全部指标

下列命令读取已交付训练状态，重新加载42个选中模型并核对正常预测入口输出。统计部分独立重算每电芯、每协议、分时段、速率、EOL与固定配对区间。

```powershell
python scripts/verify_unified_seed42_results.py --data data/unified_matr_seed42 --run-root results/unified_matr_seed42/runs --analysis-root results/unified_matr_seed42/analysis --output runtime/unified_seed42_verification.json --device cuda
```

每个单元的`COMPLETE.json`绑定配置、训练标准化参数、150轮日志、测试前选权证书、选中权重、最终可恢复训练状态和测试输出。根`MATRIX_COMPLETE.json`绑定完整运行注册。独立验证同时读取这些指纹，并重算训练预算与验证集first-min选权规则。

随附正式分析状态为`ANALYZED_WITH_REPRODUCTION_FAILURE`，独立核验为`VERIFIED_WITH_REPRODUCTION_FAILURE`。42个已保存选中模型的推理重放和独立统计通过，M2与M3-NR整程训练逐值复现为0/6。命令退出码0表示执行完成，完整状态与已知缺陷应读取JSON回执及最终报告。

核验范围包括保存日志中的150轮预算和最终优化器步数，以及选中checkpoint的验证误差重放。每轮全部历史权重未逐份保存，验证日志与最终状态的方式应据此解释。模型推理重放使用同一注册实现，独立统计代码另行计算指标。

## 从头复现唯一seed42矩阵

使用新的输出目录执行完整矩阵。标准化在各折训练电芯内拟合，权重由各折验证集完整轨迹cell-mean MAE选择，所有单元跑满150轮。

```powershell
python scripts/unified_matr/train_seed42.py --data data/unified_matr_seed42 --run-root runtime/unified_seed42_reproduction --device cuda
python scripts/unified_matr/analyze_results.py --data data/unified_matr_seed42 --run-root runtime/unified_seed42_reproduction --output runtime/unified_seed42_reproduction_analysis
python scripts/verify_unified_seed42_results.py --data data/unified_matr_seed42 --run-root runtime/unified_seed42_reproduction --analysis-root runtime/unified_seed42_reproduction_analysis --output runtime/unified_seed42_reproduction_verification.json --device cuda
python scripts/render_unified_seed42_results.py --analysis-root runtime/unified_seed42_reproduction_analysis --output runtime/unified_seed42_reproduction_figures
```

训练入口支持同一合同的原子断点恢复，并拒绝源码、冻结数据或设置不一致的恢复。独立输出目录会保留完整新执行记录，既有42个交付单元继续由其完成证书绑定。

## 从原始来源重建的范围

仓库交付冻结派生数据、来源URL与SHA、元数据审计、恢复和处理脚本。原始174份CSV约21.84 GB，四份MAT约10.87 GB，另有174份Arbin Metadata与174份控制协议。恢复下载记录和原始字节保存在本机私有来源目录。源恢复及全量逐行审核需这些档案，普通模型复核直接消费仓库内冻结数据。

源数据位置通过各恢复脚本命令行参数指定。真实老化轴、续测关系、b1c18表征段和原始Step标签例外的最终处理均有冻结的身份映射、角色表和字段级原因。数据证书包含处理源码及相关配置指纹，重新定义测量规则后应产生新的独立数据版本。

## 优势与风险

随附派生数据和完整训练状态可以交叉核对输入、预算、选权、预测与汇总统计。单seed未覆盖训练随机性，固定划分的bootstrap未覆盖重分割和重训练，历史队列暴露限制外部推广结论。旧解析Flow组保留原式，其趋势缺少结构性单调保证。正速率积分组的单调约束及旧Flow实际上升情况见最终独立核验和结果报告。
