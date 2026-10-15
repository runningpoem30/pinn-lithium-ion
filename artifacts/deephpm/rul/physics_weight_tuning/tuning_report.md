# DeepHPM RUL physics-weight tuning

Test batteries were not evaluated.

| Weight | Epoch | Validation macro RMSE | Validation RMSE | Residual RMSE | dR/dcycle vs -1 MAE |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 25 | 113.776 | 151.960 | 0.0982409 | 0.7472 |
| 0.01 | 25 | 113.737 | 151.928 | 0.0972688 | 0.7474 |
| 0.1 | 25 | 113.309 | 151.548 | 0.0891325 | 0.7493 |
| 1 **selected** | 25 | 111.274 | 150.070 | 0.0575862 | 0.7504 |
