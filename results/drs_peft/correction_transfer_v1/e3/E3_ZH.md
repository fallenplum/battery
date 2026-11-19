# 实验 3 源工况留出上的修正可预测性

源域同一折内逐个留出 Query 协议组，合计 14 个不同协议。冻结参考分别来自三个源折。每次只用同折其他组拟合，且把留出协议的 Support 和留出 Query 电芯排除于模型拟合和标准化。共核对 124 个方法、窗口与协议组组合，其中 120 个可拟合留出预测。源内条件在前期试验中有历史使用，本诊断为回顾性验证。

方向有利用 $\langle e,d\rangle<0$。全步有利用 $2\langle e,d\rangle+\|d\|^2<0$。方向有利时，MSE 的正缩放有利范围为 $0<\eta<-2\langle e,d\rangle/\|d\|^2$。两项分别用每个时距的训练组先验或固定 L2 Logistic 预测。Brier 先在评估协议组内平均，再对组取平均。单个电芯会被多个 Support 反复打分。窗口支持性参与评分范围确定，未来 mask、观测数与协议不进入任何预测特征。

在完整窗中，同时满足组宏平均 Brier 降低和多于一半源折同向的对照为 无。这些是该轮稳定性线索，模型与超参数全程固定。

## 留出预测评分

| 方法 | 时距 | 目标 | 预测量 | Brier | 比训练组先验改善 | 改善折数 | 比表征距离增量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ABBA_R4X4 | 1001_plus | full_step_helpful | physical | 0.000000 | +0.000000 | 0/1 | +0.000000 |
| ABBA_R4X4 | 1001_plus | full_step_helpful | physical_and_representation | 0.000000 | +0.000000 | 0/1 | +0.000000 |
| ABBA_R4X4 | 1001_plus | full_step_helpful | representation | 0.000000 | +0.000000 | 0/1 | +0.000000 |
| ABBA_R4X4 | 1001_plus | initial_direction_helpful | physical | 0.000000 | +0.000000 | 0/1 | +0.000000 |
| ABBA_R4X4 | 1001_plus | initial_direction_helpful | physical_and_representation | 0.000000 | +0.000000 | 0/1 | +0.000000 |
| ABBA_R4X4 | 1001_plus | initial_direction_helpful | representation | 0.000000 | +0.000000 | 0/1 | +0.000000 |
| ABBA_R4X4 | all | full_step_helpful | physical | 0.198445 | -0.046853 | 1/3 | -0.025597 |
| ABBA_R4X4 | all | full_step_helpful | physical_and_representation | 0.202795 | -0.051203 | 1/3 | -0.029948 |
| ABBA_R4X4 | all | full_step_helpful | representation | 0.172847 | -0.021255 | 1/3 | +0.000000 |
| ABBA_R4X4 | all | initial_direction_helpful | physical | 0.179105 | -0.032335 | 1/3 | -0.007972 |
| ABBA_R4X4 | all | initial_direction_helpful | physical_and_representation | 0.191629 | -0.044859 | 1/3 | -0.020496 |
| ABBA_R4X4 | all | initial_direction_helpful | representation | 0.171133 | -0.024363 | 1/3 | +0.000000 |
| LORA_R8 | 1001_plus | full_step_helpful | physical | 0.071429 | +0.000000 | 0/1 | +0.000000 |
| LORA_R8 | 1001_plus | full_step_helpful | physical_and_representation | 0.071429 | +0.000000 | 0/1 | +0.000000 |
| LORA_R8 | 1001_plus | full_step_helpful | representation | 0.071429 | +0.000000 | 0/1 | +0.000000 |
| LORA_R8 | 1001_plus | initial_direction_helpful | physical | 0.000000 | +0.000000 | 0/1 | +0.000000 |
| LORA_R8 | 1001_plus | initial_direction_helpful | physical_and_representation | 0.000000 | +0.000000 | 0/1 | +0.000000 |
| LORA_R8 | 1001_plus | initial_direction_helpful | representation | 0.000000 | +0.000000 | 0/1 | +0.000000 |
| LORA_R8 | all | full_step_helpful | physical | 0.242283 | -0.037858 | 1/3 | -0.017438 |
| LORA_R8 | all | full_step_helpful | physical_and_representation | 0.237943 | -0.033517 | 1/3 | -0.013097 |
| LORA_R8 | all | full_step_helpful | representation | 0.224845 | -0.020420 | 1/3 | +0.000000 |
| LORA_R8 | all | initial_direction_helpful | physical | 0.198445 | -0.046853 | 1/3 | -0.025597 |
| LORA_R8 | all | initial_direction_helpful | physical_and_representation | 0.202795 | -0.051203 | 1/3 | -0.029948 |
| LORA_R8 | all | initial_direction_helpful | representation | 0.172847 | -0.021255 | 1/3 | +0.000000 |

1001 圈以后共有 3 颗不同物理 Query 有观测。其中 1 折、2 颗物理 Query、2 个协议的合法训练组能产出留出预测。远期结果需结合这些分母解释。

- LORA_R8 的 201_500 窗使用 physical_and_representation 预测 full_step_helpful，Brier 改善 +0.000657，2/3 折同向。
- LORA_R8 的 501_1000 窗使用 physical 预测 full_step_helpful，Brier 改善 +0.009424，2/3 折同向。
- LORA_R8 的 501_1000 窗使用 physical_and_representation 预测 full_step_helpful，Brier 改善 +0.006808，2/3 折同向。
- LORA_R8 的 501_1000 窗使用 representation 预测 full_step_helpful，Brier 改善 +0.032130，2/3 折同向。

## 协议与物理信息诊断

| 方法 | 同协议 | 电芯对 | 有利方向率 | 全步有利率 | 初始有利但全步未改善 |
| --- | --- | --- | --- | --- | --- |
| ABBA_R4X4 | False | 153 | 0.797 | 0.765 | 5 |
| ABBA_R4X4 | True | 23 | 0.565 | 0.565 | 0 |
| LORA_R8 | False | 153 | 0.765 | 0.621 | 22 |
| LORA_R8 | True | 23 | 0.565 | 0.565 | 0 |

11/14 项物理描述在 8 位小数处有变化。电流来自可用的早期放电 VIQ 网格，覆盖不到网格以外的激励。电压范围可能因预处理采用共同网格而接近常数。容量响应来自观测电压窗口内的 Q 范围与早期容量锚点。局部形变比较相对容量坐标上的早晚电压，其中已按每条曲线的可测 Q 窗口归一化。表征距离使用冻结的 64 维编码。逐描述量的近远分层阈值仅从该轮训练对估计。描述的关联与源域预测结果共同保留，置信度受协议与电芯数限制。

## 对齐后仍变差的例子

- ABBA_R4X4 的一个完整窗例子来自折 2。MATR_b2c5 学得的方向对 MATR_b3c0 有利，正缩放上界为 0.2157，全步 MSE 变化为 +1.11568e-07。
- LORA_R8 的一个完整窗例子来自折 2。MATR_b1c21 学得的方向对 MATR_b3c8 有利，正缩放上界为 0.05218，全步 MSE 变化为 +0.000859668。
