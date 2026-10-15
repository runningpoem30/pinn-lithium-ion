# Final DataDrivenNN RUL results

Checkpoints were selected by battery-macro validation RMSE. The final
test batteries were evaluated once per seed without prediction clipping.

| Seed | Best epoch | Macro MAE | Macro RMSE | Cycle RMSE | R2 | Bias |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 52 | 194.304 | 242.492 | 315.861 | 0.31848 | -187.578 |
| 2345 | 39 | 200.517 | 244.133 | 323.124 | 0.28678 | -198.710 |
| 3456 | 45 | 206.678 | 254.337 | 340.928 | 0.20602 | -229.089 |

## Aggregate (mean +/- sample standard deviation)

- mae: 230.570285 +/- 10.204444
- rmse: 326.637508 +/- 12.897467
- r2: 0.270429 +/- 0.057987
- bias: -205.125530 +/- 21.486059
- signed_error_p05: -709.031116 +/- 13.141700
- signed_error_p50: -150.515935 +/- 11.237171
- signed_error_p95: 78.158479 +/- 30.836635
- absolute_error_p50: 153.617752 +/- 8.113094
- absolute_error_p90: 536.487615 +/- 31.741647
- absolute_error_p95: 709.031116 +/- 13.141700
- absolute_error_p99: 1031.479989 +/- 37.663493
- battery_macro_mae: 200.499497 +/- 6.187304
- battery_macro_rmse: 246.987265 +/- 6.417541
- battery_macro_bias: -173.684814 +/- 17.788608
- negative_prediction_fraction: 0.008495 +/- 0.004145
