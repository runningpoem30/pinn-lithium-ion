# Limitations and interpretation guardrails

- Only three random seeds were run, so uncertainty estimates are descriptive rather than inferential.
- Evaluation uses one fixed 40-battery held-out cohort from the same dataset; external-dataset generalization was not tested.
- RUL ground truth follows the audited operational label RUL_Flt = cycle_life - cycle. The legacy intended 80% SoH threshold was not persisted consistently in preprocessing.
- RUL rankings differ between cycle-weighted and battery-weighted metrics; neither should be presented as a universal winner.
- All RUL models underpredict long-horizon remaining life, producing substantial negative bias and weak early-life accuracy.
- A small PINN residual does not prove that the learned dynamics are physically correct or improve held-out prediction.
- Verhulst RUL residuals use PCL/cycle, whereas DeepHPM RUL residuals use cycles/cycle; their numerical magnitudes cannot be compared directly.
- DeepHPM RUL learned dynamics did not recover the audited ideal derivative dRUL/dcycle = -1, revealing an identifiability limitation.
- The comparison describes frozen test performance only. It must not be used for further model or hyperparameter selection.
