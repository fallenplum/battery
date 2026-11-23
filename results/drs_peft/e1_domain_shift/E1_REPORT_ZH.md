# E1 Bridge 输入域偏移审计

本轮只读取 prefix 输入。未来标签读取计数为 0。MICH 使用最终31颗 cache，窗口 3.0–4.15V。RWTH 使用最终40颗 cache，窗口 3.5–3.83V，8颗因 Q(V) 支持不足排除。

## 域分类器 AUC

| Domain | Cells | AUC |
|---|---:|---:|
| MATR | 169 | 1.000 |
| HUST | 77 | 0.974 |
| CALCE | 12 | 1.000 |
| SNL | 28 | 0.977 |
| MICH | 31 | 1.000 |
| RWTH | 40 | 1.000 |

对 MATR 的整体 Mahalanobis/CORAL 为 HUST 426.25/1.04、CALCE 654.96/3.88、SNL 674.14/38.60、MICH 609.79/1.37。域分类器和协方差距离均显示输入域偏移很强。

这只证明输入分布不同。它不能单独证明 Bridge 负迁移由某个通道、某种物理机制或某个模块造成。

## Native response 与 fusion12

MATR 的 response mean norm/std mean 为 9.301/0.0223，fusion12 为 8.092/0.0559。RWTH 的对应值为 8.266/0.0406 和 7.986/0.1499。RWTH 的 fusion12 输出离散度明显更大。SNL fusion12 std mean 为 0.0898，也高于 MATR。

逐通道均值偏移最大的通道包括 HUST 的 8、2、1、6，CALCE 的 8、1、2、6，SNL 的 1、8、6、2，MICH 的 1、0、8、2。通道编号仍需结合固定 12 通道语义解释。

## 同构性边界

MATR/HUST 使用冻结原生 Q(V) cache。CALCE/SNL/MICH/RWTH 使用供应 pkl 的 active-discharge 段和各自窗口重建 12 通道。电压窗口、电流规则、参考曲线和坐标认证存在差异，因此本审计只能把输入偏移作为根因候选证据，不能写成已证实因果根因。

