# Final DeepHPM RUL results

RUL is predicted directly in cycles; G and the residual have units cycles/cycle.

| Seed | Epoch | Macro RMSE | RMSE | R2 | Bias | Residual RMSE | dR/dcycle vs -1 MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 56 | 230.695 | 309.988 | 0.34359 | -211.472 | 0.116037 | 0.9735 |
| 2345 | 47 | 233.662 | 321.859 | 0.29235 | -226.680 | 0.052092 | 0.9047 |
| 3456 | 56 | 268.164 | 359.755 | 0.11591 | -268.309 | 0.055632 | 0.8753 |

Aggregate macro RMSE: 244.173 +/- 20.829 cycles.
