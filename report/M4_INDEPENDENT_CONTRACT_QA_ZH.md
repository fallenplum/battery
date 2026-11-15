# M4完整方法独立实现审查

## 范围与结论

**最终冻结源的CPU与RTX4060 CUDA各88项工程检查通过。** 检查读取同一份冻结MATR169输入、标签与split指纹，仅对训练角色前缀执行真实前向与反向。全部检查的优化器更新数为0，测试评分次数为0。CUDA增加12次fresh seed42真实训练首批的完整前向、loss及全梯度hash检查，12次完全一致。正式六单元的预测结果由训练与冻结评分另行生成。

首轮80项检查通过后，真实训练模式smoke发现首批梯度hash微差。仅修复dynamic scan仍出现间歇差异，随后M4专属算子统一封装全部scan的跨设备前向与反向，保留旧源与数学公式。最终检查使用四线程、train/dropout及原full-loss路径，两份`_FINAL.json`收据绑定该修复后的源。原80项检查使用单线程eval路径，其覆盖缺口与所有失败记录保留。

合同文件为`research/M4_FULL_METHODOLOGY_RUN_ONLY_ZH.md`，SHA256为`94ac2388411306e162243ced735188c373c03a6daabbe1cf49e9ae94c7136523`。本检查独立调用新模型正常入口，另以显式逐圈标量递推验证动态持续性值与梯度。所有被测依赖SHA绑定在两份JSON收据中，检查执行前后逐文件一致。

## 已执行证据

| 项目 | 真实检查结果 |
|---|---|
| 动态持续性 | 包含首段缺失、中间缺失、不连续真实cycle坐标与全缺失电芯，FP64 dense scan与逐圈递推最大误差4.44e-16 |
| 梯度oracle | phi梯度最大误差1.11e-16，rho梯度最大误差4.42e-15，invalid输入梯度为0 |
| M4专属SSD | 独立逐时递推前向误差4.44e-16，x/delta/A/B/C/skip全部梯度oracle通过，最大误差2.84e-14 |
| rho输入 | 172维，逐项核current physics、previous-valid global与mask、质量32、policy与mask、实际gap及稳健a/b漂移 |
| 稳健漂移 | 使用冻结global列2/11的因果median，除以上一有效圈实际gap，first-valid无上一圈信息 |
| 质量与因果性 | 核局部残差支持、feature支持、noise代理与mask，后续前缀扰动对较早rho及持续状态无影响 |
| B真实写入 | 捕获两层history的实际卷积输出，验证B来自SiLU后split，并捕获传入dense_ssd的Bprime |
| 新分支梯度 | W_B零初始化且无bias，首批两层W_B梯度L2分别0.001912与0.000698 |
| 受控写入作用 | 固定手工W_B扰动产生非零B变化，正常预测最大变化2.09e-7，rho获得非零梯度5.87e-6 |
| invalid控制 | invalid圈的B调制为0，任意极值payload保持正常预测相同，五类原始/物理输入的invalid记录梯度为0 |
| 物理坐标 | 支持的prefix cycle为整数1–100且严格递增，越界与非整数拒绝 |
| rate与积分 | 固定每整数cycle梯形积分，常速率解析积分通过，物理rate为per1000输出除1000 |
| 查询一致性 | 同一t单独、联合、乱序查询的趋势、积分、rate、最终预测均一致，所测CUDA差值为0 |
| 观测层 | raw trend在100精确等于prefix anchor，趋势单调，rate正值，residual幅度≤0.03，最终SOH界限通过 |
| 训练loss | 独立数组算术复核cellmean trajectory MAE、masked per-cell rate auxiliary乘1000、权重0.1与penalty权重0.001 |
| 数据与预算 | 冻结169电芯守卫通过，五fold各2100更新，protocol单元1950更新，合计12450 |
| 历史源文件 | 旧model/operators/trainer/config与合同基线c81aba一致，比较时仅统一Git工作树换行编码 |
| 训练首批重复 | CUDA12次fresh seed42，四线程train模式、8条训练prefix、完整future query及loss，预测/loss/全梯度hash逐次完全一致 |

## 量纲与实现选择

物理损失速率单位为SOH/cycle，使用`rate_per_1000_cycles / 1000`换算。初始速率`0.1/1000`沿用旧rate head预先固定的数值尺度，未读取测试误差进行调参。固定积分网格与查询集合无关，以100为锚点，整数区间内采用累计梯形积分的线性插值。

持续性penalty维持旧定义，按有效转换逐通道计算`((z_now-z_previous)/actual_gap)^2`，再对转换数与64通道求均值。首个有效圈不产生转换penalty。新动态rho改变状态值与梯度来源，该penalty的量纲保持兼容，沿用系数0.001。

W_B零初始化使初始Bprime与B完全相同，首批反向验证W_B自身梯度。随后采用固定手工权值进行工程扰动，证明后续动态rho与真实SSM写入之间存在可微连接。该过程没有拟合数据或训练候选消融模型。

## 优势与风险

**优势**是标量递推及梯度oracle、实际卷积/SSD张量捕获、正常模型入口与真实冻结train-prefix相互交叉验证，避免只核诊断字段文字。

**风险与范围限制**包括工程控制覆盖有限输入，正式长时间优化仍可能出现当前未触发的数值异常。W_B受控扰动的预测作用较小，只证明计算图连通性，效应大小需要正式训练后报告。尚未重复整套六单元训练，单seed结果的训练随机性仍未知。历史169电芯暴露事实继续披露。

置信度评级为**高**，限上述实际执行且SHA绑定的实现合同与数值控制。

## 复现与异常留存

入口为`scripts/verify_m4_method_contract.py`，分别传入`--device cpu`或`--device cuda`，另提供`--data data/unified_matr_seed42`和`--output`收据路径。

- `results/unified_matr_methodology_full_seed42/INDEPENDENT_CONTRACT_QA_CPU_FINAL.json`
- `results/unified_matr_methodology_full_seed42/INDEPENDENT_CONTRACT_QA_CUDA_FINAL.json`
- `results/unified_matr_methodology_full_seed42/INDEPENDENT_CONTRACT_QA_CPU.json`与同名CUDA文件，保留初版80项收据
- `results/unified_matr_methodology_full_seed42/INDEPENDENT_CONTRACT_QA_CPU_ATTEMPT1.json`
- `results/unified_matr_methodology_full_seed42/INDEPENDENT_CONTRACT_QA_CPU_CUMSUM_FIX.json`，保留仅dynamic scan修复期间CPU81项收据
- `results/unified_matr_methodology_full_seed42/INDEPENDENT_CONTRACT_QA_CPU_FINAL_ATTEMPT1.json`

首次CPU检查在65项通过后，验证脚本读取配置时使用Windows默认GBK，发生UTF-8解码异常。脚本改为显式UTF-8，保留首次失败收据，随后CPU与CUDA完整检查通过。被审模型与runner源码在该修复中保持原SHA。

最终版首次CPU检查因本次全部scan改动使SSD捕获同时覆盖局部8步与history100步路径，旧测试spy将局部B与history诊断相配，发生维度异常。验证脚本改为捕获实际100步history调用，保留该失败收据，重新完成CPU/CUDA各88项检查。模型与runner源码在该测试修正中保持原SHA。

## 最终绑定版本

| 文件 | SHA256 |
|---|---|
| methodology_full.py | a1194adab453b6a9bd03dce5211b7f1fa9ccd6eed66f333695941ef03284bdee |
| methodology_operators.py | 341cd9cdcc74bb587c556e781f8b9625e97c715f04dec27d20bdf66b2ea52967 |
| train_methodology_full.py | 67d52db2eedcdc2320d0147a92aa8c1c866f80baae24025e78884e3735cbdc79 |
| verify_m4_method_contract.py | 770c3c92dded38ba38e7d549da311fc799258fe3746f606896ed142d1280510d |

CUDA12次首批完整梯度hash均为`ce0c3c8192e0a9cba3814033410d8a02f9832657c49e12c6cf44af542d6e4f4c`，对应训练loss为0.05169570818543434。
