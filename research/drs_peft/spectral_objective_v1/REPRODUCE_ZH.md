# 复现对齐目标与步长的源域实验

从 battery 仓库根目录执行。需要父研究 `87ad1c8c403503f15833f825be91c748bbf3a7bd` 对应的 MATR 冻结缓存、Moirai 权重、三个 `reference_seed=42` 参考与原前缀描述缓存。默认从 `runtime/drs_peft/data_v1`、父研究对应的参考目录及 `runtime/drs_peft/spectral_rotation_v1` 读取。输入的精确 SHA-256 见 `RUN_IDENTITY.json`、`REGISTERED_EPISODES.json` 与父版数据锁。来自父版的文件均只读。

```powershell
$objectivePython = 'D:/Anaconda/envs/tslib/python.exe'
& $objectivePython -m scripts.drs_peft.run_spectral_objective --phase register
& $objectivePython -m scripts.drs_peft.run_spectral_objective --phase dev
& $objectivePython -m scripts.drs_peft.verify_spectral_objective --phase A
& $objectivePython -m scripts.drs_peft.plot_spectral_objective
```

Stage A 按 mapper 完整身份保存私有权重、逐 episode 记录、验证全网格预测和 SHA-256。再次运行时，同一身份且通过哈希校验的训练单元会复用。所有验证结果与真实网格工件通过折级回执关联。阶段内每电芯的验证预测先用全部有效观测算误差，再对电芯和 Support 组合等权。

Stage A 通过后，`--phase outer` 执行锁定重复与源外层评分。入口读取 Stage A 判定，仅在验证门槛通过后进入重复。所有重复和等步长控制冻结后才允许逐折读取其源 outer 角色的未来观测。已有父研究历史 outer 结果不参与选参。可用 `--phase run` 依合同顺序连续执行。完成外层评分且独立认定 Stage A 通过后，使用 `verify_spectral_objective --phase C` 重放最终评分。

Stage B 的额外元训练轨迹与开销见 `stage_b/META_TRACE.csv` 与 `stage_b/COSTS.csv`。参数谱可用 `python -m scripts.drs_peft.audit_spectral_objective_frames` 做锁后 CPU 审计，只复算真实与置乱描述下的正交基底。

所有 `.pt` 和 `.npz` 均在本地 `runtime/drs_peft/spectral_objective_v1`。mapper 的源训练轨迹、开销和各验证全轨迹预测的相对路径与哈希位于相应 `stage_a/fold*/COMPLETE.json`。公开报告与 CSV 只保留指标、登记角色、回执和路径身份。新启动且需要独立身份时，可显式传入 `--public`、`--private`，使用新的目录。私有文件可从当前受控工作区按回执校验或由整套命令重建。

正式运行使用 NVIDIA GeForce RTX 4060 Laptop GPU。设备、PyTorch 版本和本轮源代码哈希记录在运行身份。首次执行应保留源码与数据锁的完整版本。分折开发模式 `--phase dev --fold N` 只运行对应折，全部折完成后用 `--phase dev` 生成共同 Stage A 判定。
