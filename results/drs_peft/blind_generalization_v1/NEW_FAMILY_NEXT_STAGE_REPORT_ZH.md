# 新 family 泛化筛选下一阶段汇总

## 已完成

1. CALCE/SNL 输入和标签资格审计。
2. CALCE/SNL 四窗口误差与逐电芯残差诊断。
3. CALCE/SNL 的 Warmup、LoRA-12、Full FT 回顾性补充比较。
4. MICH/RWTH 的三方法主矩阵和三种回顾性补充矩阵预先锁定。
5. 使用前 100 圈摘要做 Bridge 相对收益的事后可预测性诊断。

## 当前结果

| Family | WARMUP_ONLY | LORA_12 | FULL_FT_TUNED | HEAD_ONLY | HEAD_LORA | HEAD_BRIDGE_LORA |
|---|---:|---:|---:|---:|---:|---:|
| CALCE | 0.173715 | 0.173640 | **0.147406** | 0.146751 | 0.148454 | **0.142376** |
| SNL | 0.082939 | 0.082879 | **0.069280** | 0.069270 | 0.070317 | **0.065604** |

前三列是公开标签后的 retrospective supplement。后三列是固定三方法的新 family blind 评价。

## 四窗口结论

- CALCE 中 `HEAD_BRIDGE_LORA` 在 101–200、201–500、501–1000、1001–5000 四个窗口均最低。
- SNL 中 `HEAD_BRIDGE_LORA` 在前三个窗口最低。1001–5000 只有 5 颗有效电芯，`HEAD_ONLY_TRAIN` 和 `HEAD_LORA` 更低。
- CALCE 和 SNL 的正残差主要出现在中远期窗口，说明跨 family 迁移的主要问题是退化轨迹跟随不足，而不是前缀短期拟合不足。

## 尚未执行

MICH 和 RWTH 只完成 prefix-only 资格审计和矩阵锁定，尚未构建 cache，也未读取 100 圈后的目标标签。它们保留为下一轮确认性泛化证据。

## 研究判断

当前证据支持保留 `HEAD_BRIDGE_LORA` 作为总体优先候选，但 SNL 化学体系分组方向不一致，尚不能把 Bridge 的收益归因于所有电芯都共享同一结构优势。前 100 圈摘要对 CALCE 的 Bridge 收益有小样本拟合信号，对 SNL 的留一 R² 为负，尚不能启动动态状态条件化方法。下一步应使用 MICH/RWTH 的锁定矩阵复核。

