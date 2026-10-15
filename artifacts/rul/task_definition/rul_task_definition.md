# Frozen RUL task definition

## Decision

The authoritative target is the existing `RUL_Flt` value in physical cycles:

`RUL(cycle) = legacy cycle_life - current cycle`

The exact operational endpoint is the existing `cycle_life` metadata used by
`ProcessData.m`. `LoadData.m` separately documents the intended endpoint as
the first raw discharge capacity strictly below 0.88 Ah—equivalent to SoH <
0.80 at 1.1 Ah nominal capacity—but does not assign its calculated `bat_label`
back to `cycle_life`. Therefore `RUL_Flt` remains authoritative and is **not**
redefined from processed SoH or from the final recorded cycle.

## Audit findings

- Records: 99,281
- Batteries: 124
- Label range: 0–2235 cycles
- Endpoint row present: 43 batteries
- Endpoint one cycle beyond saved rows: 81 batteries
- Processed SoH visibly below 0.80: 13 batteries
- Right-censored labels under this protocol: 0
- Every battery has integer, nonnegative RUL that decreases exactly with cycle.

The processed SoH curve is not used to recreate the endpoint because its
outlier correction and row retention do not reproduce the source metadata
crossing for most batteries. Forcing the last row to RUL 0 would alter
78,933 labels.

## Official battery splits

- train: 41 batteries, 27,584 records, from `train_ind`
- validation: 43 batteries, 30,457 records, from `test_ind`
- test: 40 batteries, 41,240 records, from `secondary_test_ind`

## Final evaluation protocol

- Fit scalers using training batteries only.
- Select hyperparameters and checkpoints using validation batteries only.
- Primary selection metric: battery-macro validation RMSE in cycles.
- Evaluate the final test batteries once per fixed seed.
- Report battery-macro, cycle-weighted, per-battery, per-seed, and ensemble results.
- Do not clip negative predictions for primary metrics; report their count.
- Align every model by `(battery_id, cycle)` and require identical targets.

## Source evidence

- `LoadData.m:80-88` computes the documented 0.88 Ah `bat_label`.
- `ProcessData.m:37-45` truncates rows beyond `cycle_life`.
- `ProcessData.m:70` constructs `RUL_i = cycle_life - Cycles_i`.
- `functions.py:16,25` loads `RUL_Flt` as the second target column.

The legacy Case A/B/C experiment splits are not adopted because they use
small hand-picked battery sets or random cycle-level splits rather than the
official battery-wise train/validation/test partition.
