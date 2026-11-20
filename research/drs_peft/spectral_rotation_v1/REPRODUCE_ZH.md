# 复现与结果读取

在 battery 仓库根目录执行。默认使用此前冻结的 MATR 数据缓存、Moirai 权重和原三个源参考。重建参考需要父实验的原训练清单与权重。MATR 外层未来标签只在本折开发完成、三个折的模型与选择全部锁定以后读取。外层预测与条件权重保存在本地 runtime。

```powershell
$rotationPython = 'D:/Anaconda/envs/tslib/python.exe'
& $rotationPython -m scripts.drs_peft.run_spectral_rotation --phase register
& $rotationPython -m scripts.drs_peft.run_spectral_rotation --phase features
Copy-Item -LiteralPath 'runtime/drs_peft/spectral_rotation_v1/FEATURES_RECEIPT.json' -Destination 'results/drs_peft/spectral_rotation_v1/FEATURES_RECEIPT.json'
& $rotationPython -m scripts.drs_peft.run_spectral_rotation --phase dev
& $rotationPython -m scripts.drs_peft.run_spectral_rotation --phase outer
& $rotationPython -m scripts.drs_peft.verify_spectral_rotation --replay
& $rotationPython -m pytest -q tests/drs_peft/test_spectral_rotation.py tests/drs_peft/test_spectral_rotation_features.py tests/drs_peft/test_spectral_rotation_summary.py tests/drs_peft/test_correction_decomposition.py
```

`--phase run` 连续执行开发、锁定、外层评分。每个完成折经过 SHA-256 校验后复用。新运行可指定 `--public`、`--private` 和 `--features` 三个空目录。

`--replay` 验证三个折的开发锁，读取已封存的本地预测并独立复算 MAE。标签读取仅覆盖已锁定的 60 颗源外层 Query 电芯。重放不进行模型优化。

`development/fold*/VALIDATION_TRIALS.csv` 保存三折全部源选参格。 `META_TRACE.csv` 保存 Support 与训练 Query 名单、观测损失及映射梯度。 `BASIS_AUDIT.json` 记录基底构建时耗、正交性和同一参考函数核验。 `DESCRIPTOR_SCALER.json` 记录折内拟合描述的特征维度、有效率和标准化参数。

`summary/METHOD_SCORES.csv` 为先对同电芯多次 draw 平均后的物理电芯等权结果。 `PAIRED_EPISODE_SCORES.csv` 含原始差值、输出等扰动和参数等范数重新前向的各自指标。等扰动的轴为固定源锚点。 `EPISODE_SCORES.csv` 含适配样本误差、成本和原始源锚位移。 `SOURCE_VALIDATION_DECISION.json` 记录每项必要判据、未过项与筛选决策。
