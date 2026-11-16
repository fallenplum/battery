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
cd "$DRS_REPOSITORY"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$DRS_RUNTIME/vendor"
export CUDA_VISIBLE_DEVICES=5
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BATTERY_SOURCE_COMMIT=16b37da3ea990e3d47634965b131ce8845dbc027
"$DRS_PYTHON" -u -m scripts.drs_peft.train --phase pilot --backbone-dir "$DRS_BACKBONE" --device cuda --resume
```

该pilot已实际启动，执行日志与产物将同步到本地。`--resume`保留命令可读性，训练器在发现身份匹配的状态文件时自动恢复。

源验证完成使用 `SOURCE_DEVELOPMENT_COMPLETE.json`。目标Query结局按软件角色隔离，缓存整体载入内存，`target_query_labels_used=0`表示这些结局未被用于计算。最终 `COMPLETED.json`留待合同规定的最终评分与核验闭环。
