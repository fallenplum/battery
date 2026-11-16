# DRS-PEFT 实际运行记录

## 已固定的执行条件

P0训练源码在提交 `16b37da3ea990e3d47634965b131ce8845dbc027` 固定。训练器SHA256为 `ebe562930e80697e01ffd103246a6a5263f6b1123c191e9e8efca8b9474aba70`。复用仓库原有 `Codex <codex@local>` 提交身份。

Windows主文件保存在本地仓库。服务器使用独立数据盘目录与隔离环境，仅承担计算。服务器实际GPU为8张RTX A6000，当前使用一张A6000共享运行，每个阶段的训练单元串行。PyTorch为2.4.1+cu121，uni2ts纯Python源码从本机已核验包复制，官方Moirai-small权重由SHA绑定。关闭AMP与TF32，启用确定性计算，每作业2个CPU线程。

## 已执行的工程验证

下列命令在本地仓库根目录执行。

```powershell
& 'D:\Anaconda\envs\tslib\python.exe' -m pytest tests/drs_peft -q --disable-warnings --basetemp runtime/drs_peft/pytest_short
```

实际55项测试通过。默认sandbox临时路径过长曾导致2项文件创建失败，使用仓库内短临时目录后通过。恢复、冻结参数和磁盘重放证据见工程验证报告。

## P0服务器运行

以下变量指向实际独立计算目录中的仓库、环境与官方权重。认证信息保存在运行会话中。

```bash
export DRS_RUNTIME=/data2/chen/codex_battery_drs_peft_20261117
export DRS_REPOSITORY="$DRS_RUNTIME/repository"
export DRS_PYTHON="$DRS_RUNTIME/environment/bin/python"
export DRS_BACKBONE="$DRS_RUNTIME/backbone"
cd "$DRS_REPOSITORY"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$DRS_RUNTIME/vendor"
export CUDA_VISIBLE_DEVICES=5
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BATTERY_SOURCE_COMMIT=16b37da3ea990e3d47634965b131ce8845dbc027
"$DRS_PYTHON" -u -m scripts.drs_peft.train --phase pilot --backbone-dir "$DRS_BACKBONE" --device cuda --resume
"$DRS_PYTHON" -u -m scripts.drs_peft.train --phase formal --backbone-dir "$DRS_BACKBONE" --device cuda --resume
```

两阶段已全部完成。pilot为6个单元、3960次更新，formal为12个单元、15840次更新。全部权重、原始预测、日志和凭据已保存到本地，各凭据文件SHA通过。18个单元完成4060独立重放，最大预测绝对差9.5367431640625e-7。`--resume`保留命令可读性，训练器在发现身份匹配的状态文件时自动恢复。

源验证完成使用 `SOURCE_DEVELOPMENT_COMPLETE.json`。目标Query结局按软件角色隔离，缓存整体载入内存，`target_query_labels_used=0`表示这些结局未被用于计算。最终 `COMPLETED.json`留待合同规定的最终评分与核验闭环。

## P1本机运行

P0完整回传后服务器长连接中断，直连及代理重连均失败。用户确认入口保持不变，后续继续重试。P1已在本机4060串行执行，源码由提交 `04b6f79` 固定，执行配置与源85/meta30角色继续使用原登记版本。

```powershell
Set-Location 'D:\develop\patchtst\battery_ai\_srsr_physics_early_precursor_v1\battery_repository'
$env:BATTERY_SOURCE_COMMIT='04b6f79'
$drsPython='D:\Anaconda\envs\tslib\python.exe'
$drsBackbone='D:\develop\patchtst\battery_ai\.cache_drs_peft\moirai_small'
foreach ($sourceSeed in @(42,123,2026)) {
    & $drsPython -u -m scripts.drs_peft.run_p1 --backbone-dir $drsBackbone --device cuda --seed $sourceSeed --no-aggregate
    if ($LASTEXITCODE -ne 0) {
        throw "P1 seed $sourceSeed failed"
    }
}
& $drsPython -m scripts.drs_peft.run_p1 --backbone-dir $drsBackbone --aggregate-only
```

每seed登记345次真实单cell梯度前后向，主诊断315次、独立端点重算10次、次要QKVO/FFN诊断20次。优化器更新为0。逐seed实际完成状态和耗时由各自 `P1_RESULT.json` 与 `P1_PUBLIC_COMPLETE.json` 记录，三seed科学门槛由 `P1_GATE.json` 汇总。

## P2等额搜索与正式源比较

P2执行器由提交 `a0f651f` 固定，SHA256为 `59536a2b752105c1c5858ac8c9939309e39ffe64b8bb1058a626ff6067a9351c`。18个LR搜索单元已全部闭环，每个60epoch、660updates，共11880updates。所选LR与全部候选原值保存在 `P2_LR_SELECTION.json` 和 `P2_SEARCH_TRIALS.csv`。

```powershell
$env:BATTERY_SOURCE_COMMIT='a0f651f'
& $drsPython -u -m scripts.drs_peft.run_p2 --phase search --backbone-dir $drsBackbone --device cuda
```

正式66个单元已全部闭环。三个源seed分别使用独立进程运行下面对应命令，沿用共同的冻结LR选择与完整注册。每个单元120epoch、1320updates，主r4六方法、近参数core56四方法、r2/8六方法合计87120次正式更新。

```powershell
& $drsPython -u -m scripts.drs_peft.run_p2 --phase formal --formal-seeds 42 --no-summary --backbone-dir $drsBackbone --device cuda
& $drsPython -u -m scripts.drs_peft.run_p2 --phase formal --formal-seeds 123 --no-summary --backbone-dir $drsBackbone --device cuda
& $drsPython -u -m scripts.drs_peft.run_p2 --phase formal --formal-seeds 2026 --no-summary --backbone-dir $drsBackbone --device cuda
```

本机先核对两进程的真实CPU、RAM、GPU与epoch进展，再启动第三个seed。三进程现场测量GPU利用率89%、显存剩余3448MiB、宿主可用内存约4.93GB，各训练进程RSS约1.65GB。每个单元使用独占文件锁，工作进程读取同一注册与LR文件。逐单元训练及验证wall time、显存峰值、实际更新和磁盘checkpoint重放由凭据分别记录，阶段经过时间另计。

完整矩阵闭环后，协调进程已实际执行汇总命令。

```powershell
& $drsPython -m scripts.drs_peft.run_p2 --phase summarize --backbone-dir $drsBackbone --device cuda
```

原执行器生成的P2 Gate为NEGATIVE。18个搜索及66个正式单元合计99000次更新，P0与P2合计102个真实训练单元、118800次更新。P1另计1035次梯度前后向、0次更新。当前门槛终止本版本P3/P4大矩阵扩展，H2/H3与最终target Query评测保持未执行状态。源阶段完成凭据、重放及负结果分别保留。
