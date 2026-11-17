# E1–E3 Windows 运行锁修复记录

## 实际故障

fold0/seed42与fold1/seed42的warmup均完成120个epoch和960次优化更新。训练日志、优化器状态、选定权重和预测重放已经生成，完成凭据在哈希产物目录时遇到PermissionError。原因是该目录内的RUN.lock仍由Windows独占锁持有，哈希读取被系统拒绝。

两份TRAIN_STATE的next_epoch均为120，history各120条，全部已有Adam状态的step均为960。TRAIN_LOG与TRAIN_STATE.history逐值相同。原训练循环代码对应提交`114ee32d2abf3a649cc10ed4e67b0d54b49d4300`。

## 修复与恢复

新增独立入口`execute_e123.py`，将运行锁映射至ignored `_locks`目录，并对原请求路径取SHA256形成锁名。相同原请求仍映射同一个锁，产物目录可以完整读取。原runner、模型、训练器及训练身份文件继续使用冻结版本的来源摘要。每次入口调用额外保存launcher、driver、配置、锁映射和训练状态前后核验。

通过原训练器正常入口恢复两份next_epoch120的状态，仅执行封存收尾。实际新增优化更新为0，原RUN_CONFIG、SOURCE_SNAPSHOT和TRAIN_STATE文件摘要匹配恢复前记录。选定权重与预测由原重构入口重新生成并核验。旧RUN.lock作为已解锁的历史文件保留，其字节由私有完成凭据校验。

原runner源码SHA为`5e30fd70377480459a4f79771160093ab3621b94b0af589c25ee23a8669e6fef`。launcher源码SHA为`8a2ef3e10317dc105b1a974a189a163bcd1e5d88d86532e30429713aa4d70133`，修复提交为`640b5d8a5566983c4cbe57cd4cc9cd5fbf8add6a`。两项恢复凭据与0更新证据保存在`IO_LOCK_AMENDMENT.json`及`io_invocations`。

## 后续正式入口

三个工况折分别由独立进程调用下面的对应fold ID。source_commit绑定科学训练源码，补充I/O实现由执行凭据另行绑定。

```powershell
$env:BATTERY_SOURCE_COMMIT='114ee32d2abf3a649cc10ed4e67b0d54b49d4300'
& 'D:\Anaconda\envs\tslib\python.exe' -u -m scripts.drs_peft.execute_e123 --phase all --fold-ids 0 --backbone-dir 'D:\develop\patchtst\battery_ai\.cache_drs_peft\moirai_small' --device cuda
& 'D:\Anaconda\envs\tslib\python.exe' -u -m scripts.drs_peft.execute_e123 --phase all --fold-ids 1 --backbone-dir 'D:\develop\patchtst\battery_ai\.cache_drs_peft\moirai_small' --device cuda
& 'D:\Anaconda\envs\tslib\python.exe' -u -m scripts.drs_peft.execute_e123 --phase all --fold-ids 2 --backbone-dir 'D:\develop\patchtst\battery_ai\.cache_drs_peft\moirai_small' --device cuda
```

独立审计重新核对实际优化器计数、历史日志、正常选权重构、补充来源与锁映射。该修复处理执行收尾，科学性能仍依据登记的216训练单元与387外层评价判断。
