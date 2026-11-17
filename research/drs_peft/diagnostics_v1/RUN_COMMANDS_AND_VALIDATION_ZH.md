# 三项诊断运行与核验记录

## 版本与数据

父版本为 `e9794316d86b5022d647c324b383e5eb57cde35c`。新分支为 `codex/drs-peft-targeted-diagnostics-20261117`。配置登记提交为 `0f04404`，D1采集与D3正式训练实现固定于 `ea66eec`，D2最终实现和独立重放核验固定于 `45f2f7c`。D2原GPU执行源码与最终CPU重放源码均保存独立SHA。新增纯缓存汇总与审计程序另随最终交付提交保存。

源训练85颗、47工况组，源验证30颗、15工况组，种子42、123、2026。原始数据、旧Episode、旧P0/P1/P2产物继续按原字节保存。逐单元来源记录绑定官方骨干、P0 checkpoint、P1基、输入缓存、源角色、运行代码与注册配置。

## 本机与服务器

本轮重试用户确认的服务器入口，8秒TCP检查超时。本轮三个诊断使用本机 RTX4060 Laptop，显存8188MiB，Python环境为 `D:\Anaconda\envs\tslib\python.exe`。Torch为2.4.1+cu121。服务器连接凭据保留在私有会话信息中。

D1逐种子顺序收集与分析，D2先GPU测量再CPU独立重放，D3先启动seed42并检查实际进程、日志、宿主内存和GPU，再启动123和2026。单进程阶段现场可用内存约11.79GB，训练进程RSS约1.64GB，GPU利用率45%、显存占用3470MiB。三进程阶段现场GPU利用率91%、显存占用4479MiB、剩余3479MiB，三条日志的epoch与更新均推进。这些是现场读数，单元凭据另存训练wall time与显存峰值。

## 实际入口

以下命令在本地主仓库执行。

```powershell
Set-Location 'D:\develop\patchtst\battery_ai\_srsr_physics_early_precursor_v1\battery_repository'
$diagPython='D:\Anaconda\envs\tslib\python.exe'
$diagBackbone='D:\develop\patchtst\battery_ai\.cache_drs_peft\moirai_small'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'

& $diagPython -u -m scripts.drs_peft.diagnostic_specificity --registration configs/drs_peft/diagnostics_v1.json --backbone-dir $diagBackbone --device cuda
& $diagPython -m scripts.drs_peft.summarize_d1_specificity
& $diagPython -m scripts.drs_peft.cross_cell_specificity

& $diagPython -m scripts.drs_peft.diagnostic_structure --backbone-dir $diagBackbone --phase structure --device cpu
& $diagPython -m scripts.drs_peft.diagnostic_structure --backbone-dir $diagBackbone --phase sensitivity --device cuda
& $diagPython -m scripts.drs_peft.diagnostic_structure --backbone-dir $diagBackbone --phase all --device cpu --output runtime/drs_peft/diagnostics_v1/D2_CPU_REPLAY
& $diagPython -m scripts.drs_peft.verify_diagnostic_structure
```

D2的最终可读源码保留原始GPU结果与独立CPU重放的版本差别。重新运行时采用独立输出目录保存新的执行结果，使用核验程序比较原始文件。

下面三条D3命令分别在独立进程运行，各自内部按constant、absolute、relative、dual、shuffled_dual顺序完成五臂。

```powershell
$env:BATTERY_SOURCE_COMMIT='ea66eec'
& $diagPython -u -m scripts.drs_peft.diagnostic_dualview --backbone-dir $diagBackbone --seeds 42 --device cuda
& $diagPython -u -m scripts.drs_peft.diagnostic_dualview --backbone-dir $diagBackbone --seeds 123 --device cuda
& $diagPython -u -m scripts.drs_peft.diagnostic_dualview --backbone-dir $diagBackbone --seeds 2026 --device cuda
```

三个种子完成后执行以下汇总和独立核验。

```powershell
& $diagPython -m scripts.drs_peft.diagnostic_dualview --summarize
& $diagPython -m scripts.drs_peft.audit_diagnostic_dualview
& $diagPython -m scripts.drs_peft.summarize_targeted_diagnostics
& $diagPython -m pytest tests/drs_peft -q --basetemp runtime/drs_peft/diagnostics_v1/pytest_final
```

## 工程与科学计账

D3正式训练前，使用独立ignored目录运行dual/seed42的CPU两epoch检查，共22次更新。门的48个状态张量均产生非零更新，最大张量范数0.04421。实际可训练参数64609，优化器学习率为0.0003与0.001。该工程结果独立于正式15单元。

D1新增1044次autograd包含24次独立端点重算，另255次前向。D2主测量360次标量反向，CPU重放另360次，合计720次。D1与D2优化器更新均为0。D3正式新增19800次优化更新，旧static三单元复用的历史更新保留原账本。

全部86项回归测试通过，有一条上游Torch flash-attention编译提示。当前配置固定使用数学SDPA。新增诊断源码、汇总、独立审计与对应测试通过Ruff。完整日志保存在ignored runtime，公开交付保留核验结果与文件指纹。

## 保存与发布

完整选权、优化器状态、梯度数组和原始预测数组留在本地主仓库的ignored runtime。公开目录保存小型CSV/JSON、逐单元日志、来源清单、测试与中文报告。Git推送采用命令级直连覆盖失效回环代理。

父版本12430个文件、1689846426字节通过原Git blob字节核验。原P2 Gate继续为NEGATIVE。本轮属于新增源开发诊断，最终目标Query未来监督的使用量保持0。
