# DeepHPM physics-weight tuning

Selection used validation batteries only. The final test batteries were not evaluated.

| Physics weight | Epoch | SoH RMSE | SoH R2 | Residual RMSE | Data loss | Physics loss | Weighted physics | Total loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 **selected** | 21 | 0.01008843 | 0.952515 | 0.0000090899 | 0.047622 | 0.004953 | 0.000005 | 0.047627 |
| 0.01 | 21 | 0.01009092 | 0.952492 | 0.0000090021 | 0.047646 | 0.004858 | 0.000049 | 0.047694 |
| 0.1 | 21 | 0.01011761 | 0.952240 | 0.0000082404 | 0.047898 | 0.004071 | 0.000407 | 0.048305 |
| 1 | 21 | 0.01035412 | 0.949981 | 0.0000053490 | 0.050163 | 0.001715 | 0.001715 | 0.051879 |

Selected physics weight: `0.001`.

This is a single-seed hyperparameter-tuning result, not a final test result.
