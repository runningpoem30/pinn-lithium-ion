# VerhulstPINN SoH analysis

This is a read-only analysis of three frozen test-prediction files; no model was retrained.
Predictions were aligned by battery ID and cycle before ensembling.

## Ensemble test metrics

- MAE: 0.01230403
- RMSE: 0.01457027
- R2: 0.866129
- Bias: 0.01176114
- 95th-percentile absolute error: 0.02577048
- Physics-residual RMSE: 0.0001328869 PCL/cycle

## Learned physically constrained parameters

- r: 0.00370818 per cycle
- K: 0.43859227 PCL
- C: 0.01324187 PCL

## Representative batteries

Lowest, median, and highest per-battery RMSE: 96, 118, 122.

Figures are available as 300-DPI PNG plus vector SVG and PDF.
