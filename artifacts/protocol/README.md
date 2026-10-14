# Frozen SoH comparison protocol

This directory is the machine-checkable contract for comparing DataDrivenNN,
VerhulstPINN, and DeepHPMNN. Creating it loads data and validates existing
artifacts but never trains or modifies a model.

- Seeds: 1234, 2345, 3456
- Features: the eight engineered features listed in the JSON manifest
- Splits: 41 training, 43 validation, and 40 final-test batteries
- Scalers: fitted on training batteries only
- Checkpoint selection: validation data loss only
- Test use: once per seed after checkpoint restoration
- Alignment key: `(battery_id, cycle)`
- Common metrics: MAE, RMSE, R2, bias, and signed/absolute error percentiles

`soh_test_reference.csv` contains only frozen test keys and targets; it contains
no model predictions. `soh_comparison_protocol.json` contains the full schema,
split membership, scaler fingerprint, metric definitions, and artifact layout.

Regenerate and validate with `python -m src.protocol`.
