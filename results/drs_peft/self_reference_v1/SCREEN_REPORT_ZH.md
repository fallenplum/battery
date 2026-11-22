# Warmup-only、LoRA-12 与 Full-FT-12 源域参照

三种方法均从同一折内的共同 warmup 状态开始，使用 12 通道、相同 fit/tune/screen 角色和 MAE。未读取 outer 或 target 未来标签。Full-FT-12 解冻 TSFM backbone、fusion12 和 head，学习率固定为 1e-4。按 tune MAE 选择时，三个折都选择第 8 个 checkpoint。

## Screen 全轨迹 MAE

| 方法 | Fold0 | Fold1 | Fold2 | 三折均值 | 可训练参数 |
|---|---:|---:|---:|---:|---:|
| WARMUP_ONLY_12 | 0.041338474 | 0.036551768 | 0.034722695 | 0.037537645 | 0 |
| LORA_12_SAME_WARMUP | 0.041175540 | 0.036489053 | 0.034691224 | 0.037451939 | 36,864 |
| FULL_FT_12 | 0.036716446 | 0.032236867 | 0.031951207 | 0.033634840 | 13,999,369 |

## 解释

Warmup-only-12 的三折均值为 0.037537645。相同 warmup 下 LoRA-12 第 8 步为 0.037451939，改善 0.000085706。Full-FT-12 第 8 步为 0.033634840，较 warmup-only 改善 0.003902805，较同 warmup LoRA-12 改善 0.003817099。Full FT 的 tune MAE 在三个折均随 1–8 epoch 下降，第 8 个 checkpoint 在本轮上限内被选中。

这只是源域内部 screen，不能当成目标域确认。Full FT 的高容量结果是经验参照，不是理论上限。当前对照仍未解冻最终目标或读取 outer 标签。
