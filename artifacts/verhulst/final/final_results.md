# Final VerhulstPINN SoH results

The physics weight was selected using validation data only. The held-out test split was evaluated once per seed after restoring the validation-selected checkpoint.

| Seed | Epoch | MAE | RMSE | R2 | Bias | Residual RMSE | r | K | C |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 21 | 0.01368379 | 0.01595062 | 0.839563 | 0.01333198 | 0.0001437804 | 0.00391292 | 0.450192 | 0.012574 |
| 2345 | 31 | 0.01308931 | 0.01535002 | 0.851417 | 0.01254430 | 0.0001230938 | 0.00357978 | 0.431253 | 0.013658 |
| 3456 | 29 | 0.01065784 | 0.01307079 | 0.892266 | 0.00940714 | 0.0001322168 | 0.00363184 | 0.434332 | 0.013493 |

## Aggregate (mean +/- sample standard deviation)

- mae: 0.0124769764 +/- 0.0016032177
- rmse: 0.0147904767 +/- 0.0015192684
- r2: 0.8610818990 +/- 0.0276488194
- bias: 0.0117611401 +/- 0.0020763190
- signed_error_p05: -0.0025162399 +/- 0.0025685368
- signed_error_p50: 0.0119315286 +/- 0.0022604017
- signed_error_p95: 0.0265544007 +/- 0.0020925632
- absolute_error_p50: 0.0120551189 +/- 0.0021451518
- absolute_error_p90: 0.0224090119 +/- 0.0018455125
- absolute_error_p95: 0.0265544007 +/- 0.0020925632
- absolute_error_p99: 0.0341522964 +/- 0.0023832644
- physical_residual_mae: 0.0000980766 +/- 0.0000087092
- physical_residual_rmse: 0.0001330303 +/- 0.0000103673
- rate_r_per_cycle: 0.0037081811 +/- 0.0001792113
- capacity_K_pcl: 0.4385922650 +/- 0.0101630873
- offset_C_pcl: 0.0132418697 +/- 0.0005841960
