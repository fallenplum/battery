# D1–D4 实际运行入口

使用项目既有Python 3.11与PyTorch 2.4.1 CUDA12.1环境，在仓库根目录执行。GPU为RTX4060 Laptop 8GiB，正式训练串行、torch CPU线程2、deterministic=True。数据保持只读，训练前先提交源码再登记。

```powershell
python -m pytest tests/test_n012.py tests/test_d14.py tests/test_d14_physical_jacobian.py -q
python -m rootcause_d14.run --phase audit
python -m rootcause_d14.metadata
python -m rootcause_d14.run --phase register
python -m rootcause_d14.oracle --domain MATR --phase development
python -m rootcause_d14.oracle --domain HUST --phase development
python -m rootcause_d14.run --phase train --domain MATR --device cuda
python -m rootcause_d14.run --phase train --domain HUST --device cuda
python -m rootcause_d14.run --phase evaluate --device cuda
python -m rootcause_d14.oracle --domain MATR --phase test
python -m rootcause_d14.oracle --domain HUST --phase test
python -m rootcause_d14.diagnostics --domain MATR --device cuda
python -m rootcause_d14.diagnostics --domain HUST --device cuda
python -m rootcause_d14.analyze --phase metrics
python -m rootcause_d14.analyze --phase replay --device cuda
python -m rootcause_d14.report
python -m rootcause_d14.verify
```

断点恢复重复相同train命令。已完成单元先核验全部哈希后跳过。每轮恢复状态包含模型、AdamW、RNG、best checkpoint、150轮完整历史及训练probe。最终状态已写入但派生历史未写入的情况同样可恢复。工程烟测在独立目录验证2epoch中断与连续运行位级一致，28次真实工程更新单列。

重新运行原测试会额外产生工程优化更新，应单列其计数。audit与register属于新执行根目录的预注册步骤，已封存执行请直接核验，修改源码需要版本化新研究。全部正式源与配置SHA见SOURCE_REGISTRATION.json，旧2678个基线文件保留字节。

selected权重与恢复状态见results/D1_D4/CHECKPOINT_INDEX.csv。每个测试npz保留完整域查询网格及原始observed mask，0占位目标不能当成真值。未来oracle与profile目录均标诊断作用域，输入特征、初始化与超参数从未读取诊断参数。
