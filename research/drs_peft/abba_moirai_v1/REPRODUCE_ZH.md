# 复验 ABBA 与 LoRA、HiRA、AdaLoRA 的电池少样本比较

从 battery 仓库根目录运行。使用原研究的 `runtime/drs_peft/data_v1`、`runtime/drs_peft/onesided_fewshot_v1/references`、Moirai-small 官方权重及父实验的固定 ZERO 预测。执行入口认证冻结数据、父协议、模型参考身份、上游移植文件和锁定状态。可用 `--data-root`、`--reference-root` 和 `--backbone-dir` 显式提供输入。

```powershell
$batteryPython = 'D:\Anaconda\envs\tslib\python.exe'
& $batteryPython -m scripts.drs_peft.run_abba_moirai --phase p0
& $batteryPython -m scripts.drs_peft.run_abba_moirai --phase source
& $batteryPython -m scripts.drs_peft.run_abba_moirai --phase lock
& $batteryPython -m scripts.drs_peft.run_abba_moirai --phase target-predict
& $batteryPython -m scripts.drs_peft.run_abba_moirai --phase target-score
& $batteryPython -m scripts.drs_peft.summarize_abba
```

源和目标预测阶段可以在不同作业按折、目标种子或方法运行。目标成对比较建议令同种子的方法运行在同一 GPU 类型。所有预测及适配器状态经哈希校验后才进入评分。评分默认使用本地数据与完整目标产物。重复 Support 的请求保留在曲线上，并复用同一物理状态。

## 移植范围

- **ABBA** 按作者代码以 `torch.svd_lowrank` 初始化第一对因子，第二对使用作者实现的 Kaiming 和零因子。前向使用等价的 Khatri–Rao 分解。 [ABBA](https://github.com/CERT-Lab/abba/blob/6a3c9fcae791101043136a5e31136131291004ec/abba.py)
- **LoRA 与 HiRA** 保留作者的零增量起点。HiRA 对共同冻结有效权重逐元素调制。 [LoRA](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py) · [HiRA](https://github.com/hqsiswiliam/hira/blob/f75f0cfe73e6729f9e1aa5402c5ca5ae36a8c0e8/hira/tuners/lora.py)
- **AdaLoRA** 保留官方正交惩罚、梯度敏感度的滑动平均、三次预算递减与阈值裁剪。CUDA 确定性模式在 CPU 上计算阈值，保留原来的 cutoff 和并列处理。 [AdaLoRA](https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/loralib/loralib/adalora.py)

P0 的 Moirai 样本来自已登记的源内折。初次工程验证遇到 PyTorch 确定性模式不支持 CUDA kthvalue 的错误。修复阈值执行位置后，正式 P0 重放通过。工程试跑与正式源内及目标预算分别计量。

公共文件包括算法代码、配置、逐电芯评分、逐层奇异谱、适配摘要以及带哈希的来源信息。带原始观测值的缓存、模型权重、适配状态和逐圈预测保存在本地运行目录。数值秩以每层实际更新矩阵的最大奇异值为相对阈值，99% 能量秩与稳定秩补充表征奇异谱集中度。

## 独立核验

AdaLoRA 对照执行了固定提交的官方代码。12 层、初始平均秩 12 到目标平均秩 8 的 CPU 夹具在 30、60 步与 FP32、FP64 的四种组合中，逐步前向输出、正则、参数、阈值和分配均位级一致。将官方提交中的 `adalora.py` 和 `layers.py` 放到同一目录后，可用 `python -m scripts.drs_peft.audit_adalora_upstream --upstream-dir PATH` 复验。[AdaLoRA 位级对照](ADALORA_UPSTREAM_PARITY.json)

使用真实登记的 MATR 电芯身份和合成预测及标签演练了中断恢复。评分在预测未齐时退出，合成标签读取数为 0。恢复后 13 次不同适配生成 16 个评分槽，两组重复 draw 均复用已存状态。评分重跑复用已验证评分文件。[集成核验](P0_INTEGRATION_AUDIT.json)
