# Key findings

- VerhulstPINN has the lowest ensemble SoH RMSE (0.01457), 5.84% below DataDrivenNN.
- DataDrivenNN has the lowest cycle-weighted ensemble RUL RMSE (324.47 cycles).
- VerhulstPINN has the lowest mean per-battery ensemble RUL RMSE (241.45 cycles), showing that weighting changes the ranking.
- DeepHPMNN is strongest in early-life ensemble RUL RMSE (481.99 cycles), but DataDrivenNN is strongest late in life (72.97 cycles).
- All RUL ensembles have substantial negative bias, especially for long-horizon early-life predictions.
- Physics residual magnitude measures internal equation consistency, not prediction accuracy; the RUL residuals also have different units across PINNs.
