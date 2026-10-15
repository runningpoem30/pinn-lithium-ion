# Combined SoH and RUL physics-residual analysis

The graph uses elapsed life fraction on a shared x-axis. Shaded bands show the 10th–90th percentile.

- SoH Verhulst residual RMSE: 0.00013288695 PCL/cycle.
- SoH DeepHPM residual RMSE: 1.100236e-05 PCL/cycle.
- RUL Verhulst residual RMSE: 0.0002216901 PCL/cycle.
- RUL DeepHPM residual RMSE: 0.045193203 cycles/cycle.

DeepHPM is more internally consistent for SoH, but Verhulst predicts SoH more accurately. This demonstrates why physics residual and prediction error must be reported separately. The two RUL residual magnitudes cannot be directly compared because their units differ.
