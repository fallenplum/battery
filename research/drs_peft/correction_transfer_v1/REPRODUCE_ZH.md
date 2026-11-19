# 运行与复核

从 battery 仓库根目录运行。使用父实验的本地 raw-prefix cache、冻结参考权重、ABBA 目标预测和适配状态。输出落在独立的 `correction_transfer_v1` 路径。原始预测、适配权重和 64 维编码留在本地 runtime。命令可按既有认证产物继续运行。

```powershell
$diagnosisPython = 'D:\Anaconda\envs\tslib\python.exe'
& $diagnosisPython -m scripts.drs_peft.run_correction_replay --phase source-predict
& $diagnosisPython -m scripts.drs_peft.run_correction_replay --phase source-score --device cpu
& $diagnosisPython -m scripts.drs_peft.correction_decomposition --phase target-query
& $diagnosisPython -m scripts.drs_peft.run_correction_replay --phase target-support
& $diagnosisPython -m scripts.drs_peft.summarize_correction_decomposition
& $diagnosisPython -m scripts.drs_peft.run_correction_scale --phase run
& $diagnosisPython -m scripts.drs_peft.run_correction_scale --phase summarize --device cpu
& $diagnosisPython -m scripts.drs_peft.summarize_correction_scale
& $diagnosisPython -m scripts.drs_peft.analyze_correction_transfer
```

E1 的原始字段 `dot_ed` 和 `energy_dd` 分别对应电芯等权的交叉矩与修正能量。统计方向误差时以 `1e-14` 为容差。时距缺少观测时保留行和空值。E1 的 oracle 仅由本行对应标签复算，决策阶段无法获得。

E2 的同扰动比较将既有两个真实输出分别向其同一参考输出缩放，以使固定源锚上的输出 RMS 相等。缩放由这两个输出与参考输出决定，锚无未来标签。原幅度预测与缩放诊断分别保存。
分窗复算脚本读取本地 E2 轨迹预测及源开发标签。历史墙钟耗时来自本地执行时间回执，有该回执时写入 E2 报告。

E3 的拟合和结果读取均为源域。Support 分组与 Query 电芯是验证身份。协议匹配统计用于描述数据，并提供留出分组变量。
