# VerhulstPINN physics-weight tuning

Selection used validation batteries only; final-test batteries were not evaluated.

| Weight | Epoch | SoH RMSE | SoH R2 | Residual RMSE | Data loss | Physics loss | Weighted physics | Total loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 21 | 0.01006027 | 0.952780 | 0.0001570537 | 0.047356 | 1.478672 | 0.001479 | 0.048835 |
| 0.01 **selected** | 21 | 0.00990907 | 0.954189 | 0.0001516916 | 0.045944 | 1.379426 | 0.013794 | 0.059738 |
| 0.1 | 18 | 0.01147371 | 0.938579 | 0.0001335137 | 0.061598 | 1.068629 | 0.106863 | 0.168461 |
| 1 | 24 | 0.01928787 | 0.826430 | 0.0000802403 | 0.174072 | 0.385977 | 0.385977 | 0.560049 |

Selected physics weight: `0.01`.

This is a single-seed validation result, not a final-test result.
