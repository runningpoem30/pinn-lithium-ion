# Final DeepHPM SoH results

The final test split was evaluated once per seed after validation-based checkpoint selection.
No best seed was selected or discarded.

| Seed | Best epoch | SoH MAE | SoH RMSE | SoH R2 | Residual MAE | Residual RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 21 | 0.01421045 | 0.01658748 | 0.826495 | 0.0000114587 | 0.0000140397 |
| 2345 | 30 | 0.01439570 | 0.01663617 | 0.825475 | 0.0000089426 | 0.0000104065 |
| 3456 | 29 | 0.01136035 | 0.01377110 | 0.880412 | 0.0000086259 | 0.0000108132 |

## Aggregate (mean +/- sample standard deviation)

- soh_mae: 0.0133221633 +/- 0.0017015061
- soh_rmse: 0.0156649148 +/- 0.0016402759
- soh_r2: 0.8441275801 +/- 0.0314274427
- physical_residual_mae: 0.0000096757 +/- 0.0000015521
- physical_residual_rmse: 0.0000117531 +/- 0.0000019906
