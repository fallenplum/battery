# E1–E3 最终执行与核验

## 完成范围

本轮完成三折、三个种子的216个真实训练单元。其中warmup9个、静态核心27个、条件化180个，正式优化更新207360次。387项源外层评价全部完成，包含27个静态起点和360个条件化评价。最终目标校准次数为0，最终目标Query未来标签使用量为0。

正式训练与外层评价均使用本机RTX4060 Laptop，Torch为2.4.1+cu121，采用固定数学SDPA。三个工况折由独立进程并行执行，各单元保留实际环境、wall time和显存峰值。

主科学训练源码固定于提交`114ee32d2abf3a649cc10ed4e67b0d54b49d4300`。Windows锁映射入口对应`640b5d8a5566983c4cbe57cd4cc9cd5fbf8add6a`。科学数值汇总使用源码SHA`c8305d8f3b7d1d35d7c15bc806315f34a69d46eeb74b232188334176248a89f9`，独立公开快照保存在validation目录。

## 关键实现选择

每折从官方W0重建Bridge/Head起点，归一化与梯度基仅拟合本折fit。静态核心和条件化阶段冻结同一个Bridge/Head，各视角从相同来源的同一个C0出发。每个条件化分支仅训练1440个参数。增量半径以C0为中心，主幅度为每层W0范数的1%，Gradient-Core另含0.25%和4%。matched_norm固定已选参数，将非零增量归一至登记半径。

这是对固定起点条件化作用的隔离检验。Bridge/Head冻结方式、fit规模、新工况划分与幅度控制均与旧D3存在差别，当前结果无法分别归因这些因素。源115颗电芯具有历史暴露，结果属于重新分组后的回顾性交叉验证。

## 真实故障与恢复

首批两个warmup在完成960次更新后，因Windows独占RUN.lock导致完成凭据的哈希读取失败。通过独立锁映射入口完成封存恢复，新增更新为0，原训练状态与身份文件摘要匹配。工程两epoch检查另有16次更新，独立于正式207360次。详细记录见IO_RECOVERY_ZH.md，初始错误日志与恢复日志均保留。

## 外层评分与独立审计

216个单元及9套fit-only梯度基完成后，先执行source完整审计，再开放外层评价。三个fold的score入口各处理129项结果，并通过正常磁盘选权重构逐bit重放。score过程优化器更新为0。

独立CPU审计核验了fit/inner/outer身份、当前配置与source snapshot、实际优化器step、共同Bridge/Head、固定C0、来源基、原始监督与观察mask。74175条电芯窗口误差由私有NPZ重新计算。幅度诊断覆盖165600条电芯×层记录，球约束与范数匹配均通过，范数匹配不可达记录数为0。

报告增强只展开已冻结的逐幅度和逐折数值。原12个数值CSV/JSON字节一致，原统计执行源码SHA保留。96个展示百分数与原contrasts绑定核验，两次report-only渲染的报告及manifest字节一致。最终测试计数和完整输出位于validation/E123_QA_DELIVERY.log。

## 复现入口

所有命令在本地主仓库运行，官方权重目录为本机缓存。

```powershell
Set-Location 'D:\develop\patchtst\battery_ai\_srsr_physics_early_precursor_v1\battery_repository'
$e123Python='D:\Anaconda\envs\tslib\python.exe'
$e123Backbone='D:\develop\patchtst\battery_ai\.cache_drs_peft\moirai_small'
$env:BATTERY_SOURCE_COMMIT='114ee32d2abf3a649cc10ed4e67b0d54b49d4300'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& $e123Python -m scripts.drs_peft.audit_e123 --phase source --require-complete
& $e123Python -m scripts.drs_peft.execute_e123 --phase score --fold-ids 0 --backbone-dir $e123Backbone --device cuda
& $e123Python -m scripts.drs_peft.execute_e123 --phase score --fold-ids 1 --backbone-dir $e123Backbone --device cuda
& $e123Python -m scripts.drs_peft.execute_e123 --phase score --fold-ids 2 --backbone-dir $e123Backbone --device cuda
& $e123Python -m scripts.drs_peft.audit_e123 --phase outer --require-complete
& $e123Python -m scripts.drs_peft.summarize_e123 --render-report-only
```

数值统计的原执行源码快照与来源SHA随交付保存。当前报告渲染入口读取已封存的统计表，源训练命令见IO_RECOVERY_ZH.md。

## 保存与结论

完整权重、优化器状态、梯度缓存和原始预测数组保存在本地ignored runtime。公开交付包含代码、冻结配置、来源凭据、逐单元真实日志、逐电芯结果、完整外层对照和中文报告。

absolute和dual均未同时通过E1、E2、E3。目标推进标志为False，本轮止于源工况复核。当前结论支持重新审视状态条件化的稳定贡献，固定子空间来源、联合训练和远期预测机制仍需对应证据。
