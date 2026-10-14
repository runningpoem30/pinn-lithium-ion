# Final DataDrivenNN SoH results

The final test split was evaluated once per seed after validation-based checkpoint selection.
No physics loss or test-based seed selection was used.

| Seed | Best epoch | MAE | RMSE | R2 | Bias |
|---:|---:|---:|---:|---:|---:|
| 1234 | 21 | 0.01420933 | 0.01658647 | 0.826516 | 0.01394498 |
| 2345 | 30 | 0.01439681 | 0.01663795 | 0.825438 | 0.01415163 |
| 3456 | 29 | 0.01136136 | 0.01377253 | 0.880387 | 0.01032883 |

## Aggregate (mean +/- sample standard deviation)

- mae: 0.0133224995 +/- 0.0017009788
- rmse: 0.0156656495 +/- 0.0016396937
- r2: 0.8441138673 +/- 0.0314182728
- bias: 0.0128084793 +/- 0.0021499235
- signed_error_p05: -0.0013280263 +/- 0.0026156699
- signed_error_p50: 0.0131327709 +/- 0.0025310514
- signed_error_p95: 0.0277728925 +/- 0.0025393528
- absolute_error_p50: 0.0132028063 +/- 0.0024111242
- absolute_error_p90: 0.0235259354 +/- 0.0022593557
- absolute_error_p95: 0.0277728925 +/- 0.0025393528
- absolute_error_p99: 0.0366735222 +/- 0.0023282024
