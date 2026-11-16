# DRS-PEFT 工程验证记录

## 验证范围与状态

**最终完整工程套件已通过 61 项测试。** 本记录覆盖官方骨干与桥接、早期输入、PEFT 算子、训练恢复、源域选权和产物重放。此处记录的更新均属于工程验证，正式 P0、P1、P2 和目标域少样本实验使用各自登记的运行产物。

本次最终训练器文件 SHA256 为 `ebe562930e80697e01ffd103246a6a5263f6b1123c191e9e8efca8b9474aba70`。最终磁盘恢复复核直接调用该版本的正常 `train_source_unit` 入口。

## 完整测试套件

P0锁源时日志记录 `55 passed, 1 warning in 23.35s`。补齐P1与P2执行边界后，最终完整日志记录 `61 passed, 1 warning in 59.94s`。

| 文件 | 通过数量 | 主要覆盖 |
| --- | --- | --- |
| `tests/drs_peft/test_bridges.py` | 23 | 官方权重逐张量一致、两种桥、B1/B8、CPU/CUDA、掩码、输入与 Q 投影梯度、原生路径与官方 forward 一致 |
| `tests/drs_peft/test_battery_baseline.py` | 9 | 源域拟合物理尺度、合法 reference、旧 M1 summary 逐位一致、共享 head、全部参数梯度、CPU/CUDA dropout 重放 |
| `tests/drs_peft/test_data.py` | 8 | 实体与协议划分、Support/Query 分母、真实 early drift、canonical 单位、源域限定标准化、Stanford 后缀扰动隔离 |
| `tests/drs_peft/test_peft.py` | 21 | LoRA/core 显式矩阵等价、门控梯度、目标更新范围、序列化、LoRA-XS/GA、失败原子性、参数量、P1 梯度缓存重放、三seed门槛及P2调度与失败留档 |

首次完整套件使用 Windows 默认长临时目录时，2 项测试因临时产物路径过长写出失败。改用 `--basetemp runtime/drs_peft/pytest_short` 后，完整 55 项通过。该路径问题与后续正式训练目录分别保留记录。

完整日志保存在 `D:\develop\patchtst\battery_ai\_srsr_physics_early_precursor_v1\private\drs_peft_20261117\QA_FULL55.log`。

最终61项完整日志保存在同目录的 `QA_FULL61_FINAL.log`。两次完整套件均使用仓库内短临时目录。

## 正常训练入口与恢复

真实 MATR 工程 smoke 使用 native bridge、B0、seed42、2 epochs，完成 **22 次 optimizer updates**。所选权重经独立重建，预测重放最大绝对误差为 0。该运行早于训练器最终加固版本，最终加固版本另有以下有限合成恢复验证。

真实 smoke 收据位于 `D:\develop\patchtst\battery_ai\_srsr_physics_early_precursor_v1\private\drs_peft_20261117\real_smoke\P0\one\native_multivariate\B0_pretrained_frozen\seed42\SOURCE_DEVELOPMENT_COMPLETE.json`。收据列出的文件 SHA256 已独立复核。

使用含 dropout 的小模型和明确标记的合成标签，按同一训练入口完成以下检查。

| 检查 | 结果 |
| --- | --- |
| 连续训练 3 epochs，与 epoch1 结束后中断恢复比较 | 选择的模型张量、最佳 epoch、验证预测逐位一致 |
| 训练全部 epochs 后、finalization 前中断，再恢复 | 选择的模型张量和验证预测逐位一致 |
| 最终锁源版本实际 `torch.load` 磁盘 `CHECKPOINT_SELECTED.pt`，经正常入口独立重建 | 预测逐位一致 |
| 删除本应存在的 LoRA A 权重键 | `load_selected` 拒绝不完整状态 |
| 训练时故意修改冻结参数 | 训练前后 frozen fingerprint 检查拒绝生成完成收据 |
| 对完成收据列出的文件追加一个空格 | 正常恢复入口的文件 SHA256 检查拒绝该产物，随后恢复原始字节 |
| 相同 Windows 环境和 seed42 重建真实 B1 随机冻结模型 | 完整模型 state fingerprint 相同 |
| B0/B1 相同 seed42 | 共同 head 和 bridge fusion 初始化逐位一致 |
| M1 encoder 的 CPU/CUDA 训练态 dropout，保存并恢复 RNG | 前向逐位一致，全部可训练参数梯度存在且有限 |

第一次有限恢复与故障探针共 **24 次合成 optimizer updates**，其中三个恢复对照各 6 次、冻结参数故障探针 6 次。最终锁源版本磁盘恢复增量验证另有 **6 次合成 optimizer updates**。两批合计 **30 次**。

`test_peft.py` 的 target-delta 范围测试每次完整套件另执行 **1 次合成 optimizer update**。P1 梯度审计的两个 toy case 只计算梯度与重放，各自 `optimizer_updates=0`。真实输入的两桥与 B3 测试执行前向和反向，optimizer updates 为 0。

恢复证据保存在仓库内以下私有路径。

- `private/drs_peft_20261117/audit_train_recovery.py`
- `private/drs_peft_20261117/toy_train_recovery_v1/ENGINEERING_RECOVERY_REPORT.json`
- `private/drs_peft_20261117/toy_disk_delta_locked/ENGINEERING_DISK_DELTA_REPORT.json`

以上工程更新与正式研究更新分别记账。工程验证所用合成标签的性能数值仅服务恢复一致性断言。

## 跨操作系统的 B1 随机骨干身份

后续 Linux 训练产物在 Windows 独立重放时，实际初始张量 SHA 守卫发现 B1 在两种环境中仅凭相同 seed 重建的骨干存在差异。对 seed42 的服务器权威随机状态与本地 `make_model(..., B1_random_frozen, seed=42)` 做了全量只读逐张量比较。

| 项目 | 实测结果 |
| --- | --- |
| 比较张量总数 | 94 |
| 比较元素总数 | 13,827,528 |
| 有逐位差异的张量数 | 57 |
| 有逐位差异的元素数 | 4,878,172 |
| 最大绝对差 | 1.1920928955078125×10⁻⁷ |
| 服务器 seed42 张量 SHA256 | `633c02cdeeb0d0bc5e00d97f079c9e96ce3a7a95525c0c674165356ee6403d85` |
| Windows 相同 seed42 重建张量 SHA256 | `901e65304c173a69ed4ef413e37b8009bb568edd7eea7a70ede04b7b4e6d6e28` |

差异涉及 `mask_encoding.weight[1,384]` 中5个元素，最大差为上述全局最大值。`in_proj.weight[5,384,128]` 与 bias 的最大差为2.9802322387695312×10⁻⁸。六层 q/k/v/out 投影 `[384,384]`、fc1/fc_gate `[1024,384]` 及部分分布输出投影也存在差异，相关线性权重最大差为3.725290298461914×10⁻⁹。完整94项形状、差异计数和误差记录保存在 `private/drs_peft_20261117/B1_CROSS_OS_INITIALIZATION_COMPARISON.json`。本次比较的 optimizer updates 为0。

差异覆盖使用 normal 初始化的 embedding 以及使用 uniform/Kaiming-uniform 初始化的线性层。具体底层数学函数、编译器或 CPU 指令的因果归属为 `UNVERIFIED`。PyTorch v2.4.1 官方可复现性文档明确说明跨平台以及 CPU/GPU 之间即使使用相同 seed，逐位结果也可能不同。[PyTorch v2.4.1 官方文档源码](https://github.com/pytorch/pytorch/blob/v2.4.1/docs/source/notes/randomness.rst)

因此，B1 的跨环境重放绑定服务器训练初始化的完整随机骨干状态、文件 SHA 和张量 SHA，并在本地严格加载后复核。权威文件为 `results/drs_peft/P0/random_backbones/seed42.pt`、`seed123.pt`、`seed2026.pt`，清单为同目录 `RANDOM_BACKBONE_MANIFEST.json`。三个文件 SHA 已独立核验。seed42 文件 SHA256 为 `9a84467ba9aecc5211148fbed110a219902873d4e75fb85a7df90585d83c3307`。服务器已完成训练的同机重建断言与此跨环境来源检查各有对应产物。

## 优势与风险

优势在于恢复与选权经真实序列化和正常入口检查，冻结权重、缺键和产物篡改均有失败探针。B0/B1 的相同初值检查减少随机初始化路径引入的比较偏差。

风险在于有限恢复使用小模型与 CPU，真实 MATR smoke 只运行了 2 epochs。本次证据覆盖工程可执行性与指定防护，完整预算下的优化表现、种子稳定性、跨域可迁移性和目标域少样本效果仍需对应正式实验产物。

**工程结论置信度为高**，依据为实际测试、张量一致性断言、SHA256 验证和有限故障探针。科学效果尚待正式实验评估。
