# Recommended results slides

The package reports frozen held-out results. Test rankings are descriptive and were not used to retune or select checkpoints.

## Slide 1 — Experimental protocol and headline results

Use `01_cross_task_results_overview.png` with the compact results table.
State the battery-wise split, three fixed seeds, validation-only checkpoint/hyperparameter selection, and single final test evaluation.

## Slide 2 — SoH results

Use the SoH metric and trajectory figures. Headline: VerhulstPINN has the best SoH metrics, while improvements over the other models are modest.

## Slide 3 — Overall RUL results

Use actual-versus-predicted RUL. Headline: DataDrivenNN is best cycle-weighted; all models underpredict long remaining-life horizons.

## Slide 4 — Where RUL models succeed and fail

Use life-fraction and horizon figures. Headline: DeepHPM is strongest early, while DataDrivenNN is strongest in middle/late life.

## Slide 5 — Physics diagnostics

Use the residual figures. Do not compare Verhulst and DeepHPM RUL residual magnitudes because their units differ.

## Slide 6 — Conclusions and limitations

Use `key_findings.md` and `limitations.md`. Avoid claiming a universal winner or causal physical discovery.

## Figure captions

### Slide 1 — Cross-task result overview

Ensemble SoH and RUL metrics on the same 40 held-out batteries. Results are descriptive and were not used for model selection.

File: `figures/01_cross_task_results_overview.png`

### Slide 2 — SoH accuracy across three fixed seeds

VerhulstPINN has the lowest SoH error, but all three models remain close.

File: `figures/02_soh_metric_comparison.png`

### Slide 2 — Representative held-out SoH trajectories

Measured and predicted degradation paths with ±1 seed standard deviation.

File: `figures/02b_soh_trajectories.png`

### Slide 3 — Actual versus predicted RUL

All models underpredict long-horizon RUL; the data-driven ensemble has the lowest cycle-weighted RMSE.

File: `figures/03_rul_actual_vs_predicted.png`

### Slide 4 — RUL error across battery life

Early-life RUL is substantially harder than late-life prediction.

File: `figures/04_rul_error_across_life.png`

### Slide 4 — RUL error by prediction horizon

Prediction error rises strongly as the remaining-life horizon increases.

File: `figures/04b_rul_horizon_error.png`

### Slide 5 — SoH physics residual diagnostics

Residual consistency is a diagnostic and is not equivalent to predictive accuracy.

File: `figures/05_soh_physics_residuals.png`

### Slide 5 — RUL physics residual diagnostics

Verhulst and DeepHPM residuals are separated because their physical units differ.

File: `figures/05b_rul_physics_residuals.png`

### Slide 5 — Combined SoH and RUL physics residuals

Residuals across elapsed battery life. SoH residuals share PCL/cycle units; the two RUL panels use different units and must be interpreted separately.

File: `figures/06_combined_soh_rul_physics_residuals.png`

