# Head-aware retrospective target evaluation

## Lock and information boundary

The Head-aware target lock was created before target future labels were opened. It fixes seven methods, source seed 42, 12-channel input, source-selected states, MATR and HUST target query IDs, and K=0 zero-shot prediction. Prefixes were built from the registered early input cache. Target future labels were opened only after prediction files were locked.

The target queues are historically exposed. These results are retrospective target evaluation and do not constitute blind confirmation.

## MATR pooled target

| Method | Target MAE |
|---|---:|
| HEAD_LORA | 0.023116 |
| LORA_12 | 0.023374 |
| FULL_FT_TUNED | 0.023549 |
| WARMUP_ONLY | 0.023459 |
| HEAD_ONLY_TRAIN | 0.024103 |
| HEAD_BRIDGE_TRAIN | 0.026020 |
| HEAD_BRIDGE_LORA | 0.026183 |

## HUST C_HUST target

| Method | Target MAE |
|---|---:|
| HEAD_ONLY_TRAIN | **0.025141** |
| FULL_FT_TUNED | 0.025574 |
| HEAD_LORA | 0.027208 |
| HEAD_BRIDGE_TRAIN | 0.030912 |
| HEAD_BRIDGE_LORA | 0.030617 |
| LORA_12 | 0.061215 |
| WARMUP_ONLY | 0.061518 |

The ranking differs across target domains. Head-aware methods improve over the backbone-only methods on both queues, while Head+Bridge variants perform worse than Head-only on HUST. No multi-seed conclusion is registered.\n
