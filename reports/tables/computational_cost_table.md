| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5090 (33.7 GB VRAM) |
| CPU cores | 384 |
| Execution mode | 4 horizons in parallel |
| Total wall clock | ~12.3 hours |
| 10min horizon | 738.7 min |
| 30min horizon | 536.8 min |
| 60min horizon | 177.9 min |
| 120min horizon | 97.7 min |
| BiLSTM HPO | 10 Optuna trials/horizon, ~25-50 min each |
| Tree models | Cached after first fold |
| Inference | Sub-millisecond per sample (tree models, CPU) |

*Note: Conducted on a rented Vast.ai cloud instance.*
