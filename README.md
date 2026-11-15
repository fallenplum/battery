# Battery 研究实验记录

本仓库记录电池早期电学响应、未来退化结局与数据资格审计的代码、合同及结果。

## MATR与Stanford扩展

已补齐MATR169颗与Stanford181颗的预测实验，分别按中期和晚期实际观察范围保留有效样本。MATR中期加入Q(V)后MAE下降49.28%，电池和工况组配对区间均为正。Stanford当前联合线性模型出现分布外外推失败，原始负结果与真实前缀复核完整保留。

最新结果见[多数据集扩展报告](report/MULTIFAMILY_EXTENSION_REPORT_ZH.md)、[MATR资格](report/MATR_EXTENSION_QUALIFICATION_ZH.md)、[Stanford资格](report/STANFORD_EXTENSION_QUALIFICATION_ZH.md)及[106项独立核验](results/MULTIFAMILY_EXTENSION_QA.json)。

本轮按供应循环坐标定义预测任务，原始物理循环标定状态作为解释边界保留。完整历史HUST研究见下文。

## 当前实验

`SRSR Physics early precursor v1` 从 HUST、MATR、Stanford_2 的 427 个供应电池文件开展定向 D0 审计，核验循环坐标、曲线可比性、早期特征与未来结局支持。历史 576 电池及已完成响应研究仅用于身份映射和证据追踪。

本轮D0、探索性实验B与新增SDU审计已完成。具体结果见[完整实验报告](report/EXPERIMENT_REPORT_ZH.md)及[SDU审计](report/SDU_DATA_AUDIT_ZH.md)。

## 首轮HUST与SDU结果

- HUST官方归档77颗、146122个原始循环完成安全静态解析，供应前100记录的V/I逐点匹配。供应K100完整的76颗进入实验B，训练/验证/测试为33/11/32。
- 主目标500–1000圈平均退化速率，联合模型MAE为3.1316e-5，容量对照为3.4020e-5。配对MAE改善95%区间跨0，稳定增量收益证据有限。
- HUST77颗均未观察到80%阈值事件，EOL保留右删失。膝点算法存在光滑曲率反例，候选作为形状诊断记录。
- 新增SDU有86颗、49146圈，完整元数据与全点检查完成。9199条后续重复VIQ、原始循环/Protocol_ID缺口及真实温度支持均已记录。

## 文件

- `configs/d0_protocol.json` 保存开始执行前登记的观察窗口与资格参数。
- `scripts/run_d0.py` 执行逐电池供应信号审计及早期特征提取。
- `scripts/physics_targets.py` 生成未来斜率、EOL事件/删失和膝点候选。
- `results/d0` 保存逐电池资格、未来工况、特征和目标支持表。
- `scripts/hust_safe_pickle.py` 使用符号解析恢复对象数据，不执行pickle全局构造器。
- `scripts/build_hust_raw_targets.py` 从原始净容量计数生成未来目标并核验测量定义。
- `scripts/run_experiment_b.py` 注册、选模、封存及评价实验B。
- `scripts/audit_sdu.py` 与 `scripts/refresh_sdu_d0.py` 审计新增SDU并验证最终D0定义。
- `data/cell_metadata.csv` 为B使用的不可变供应元数据快照，`data/legacy_576_ids.json` 为历史队列身份集合。

数据源路径、私有缓存与恢复的源数据由命令行参数指定。各脚本通过`--help`提供参数。B的注册文件保留所有输入SHA、精确ID拆分、模型列和选模规则。

## B复现命令

在新的输出目录复现，已完成输出的评价入口保持封存。

```bash
python scripts/run_experiment_b.py --eligible-csv results/d0/eligible_cells.csv --features-csv results/d0/early_observed_features.csv --target-csv results/d0/HUST_RAW_CAPACITY_TARGET_SUPPORT.csv --metadata-csv data/cell_metadata.csv --output runtime/experiment_b_reproduction --register-only
python scripts/run_experiment_b.py --eligible-csv results/d0/eligible_cells.csv --features-csv results/d0/early_observed_features.csv --target-csv results/d0/HUST_RAW_CAPACITY_TARGET_SUPPORT.csv --metadata-csv data/cell_metadata.csv --output runtime/experiment_b_reproduction
```

原始数据来源及测量口径见[来源审计](report/PROVENANCE_REVIEW_ZH.md)和[原始容量目标核验](report/HUST_RAW_CAPACITY_TARGET_REVIEW_ZH.md)。后者明确区分原始计数与供应电流积分重建容量。

## 优势与风险

优势为每颗电池身份、资格与未来目标可追溯，且在建模前登记特征计算条件。风险为异构来源的循环编号、预处理时间、插值和终止条件可能不同，需要按实际核验证据限定结论。
