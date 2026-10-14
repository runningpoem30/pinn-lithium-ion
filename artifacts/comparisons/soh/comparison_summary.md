# Final SoH model comparison

This report was calculated from saved test predictions only; no model was retrained.
All nine prediction files were aligned by `(battery_id, cycle)` and verified to contain identical targets.

## Test metrics

| Model | Seed-mean MAE ± SD | Seed-mean RMSE ± SD | Seed-mean R2 ± SD | Ensemble RMSE | Ensemble R2 | Ensemble bias |
|---|---:|---:|---:|---:|---:|---:|
| DataDrivenNN | 0.013322 ± 0.001701 | 0.015666 ± 0.001640 | 0.8441 ± 0.0314 | 0.015475 | 0.8490 | 0.012808 |
| VerhulstPINN | 0.012477 ± 0.001603 | 0.014790 ± 0.001519 | 0.8611 ± 0.0276 | 0.014570 | 0.8661 | 0.011761 |
| DeepHPMNN | 0.013322 ± 0.001702 | 0.015665 ± 0.001640 | 0.8441 ± 0.0314 | 0.015474 | 0.8490 | 0.012808 |

## Relative to DataDrivenNN

- VerhulstPINN seed-mean RMSE reduction: 5.587%
- VerhulstPINN ensemble RMSE reduction: 5.845%
- DeepHPMNN seed-mean RMSE reduction: 0.005%
- DeepHPMNN ensemble RMSE reduction: 0.004%

## Interpretation guardrails

- Seed means describe repeated-training performance; ensemble metrics describe averaged predictions and are not interchangeable.
- Lower physics residual indicates closer agreement with a model's physical equation, not automatically better prediction accuracy.
- Results are cycle-weighted because every cycle is one evaluation record.
- No statistical significance claim is made from three seeds.

Representative batteries: 96, 106, 122.

Figures are saved as 300-DPI PNG and vector SVG/PDF files.
