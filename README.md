# Physics-Informed Neural Networks for Predictive Failure Analysis of Lithium-Ion Batteries

This project focuses on analysing lithium-ion battery degradation and developing machine-learning models for **State of Health (SoH)** and **Remaining Useful Life (RUL)** prediction.

The final goal is to move from a conventional data-driven neural network to a **Physics-Informed Neural Network (PINN)**, where battery degradation behaviour is incorporated into the learning process.

The project is being developed progressively rather than building the complete PINN system at once.

```text
Battery Data
     ↓
Data Processing
     ↓
Feature Engineering
     ↓
ML-Ready Dataset
     ↓
Baseline Neural Network
     ↓
Physics-Informed Neural Network
     ↓
Final Model Evaluation
```

At the current stage, the work has reached the beginning of the **baseline neural-network stage**.

---

## Dataset and problem

The project uses the public lithium-ion battery dataset associated with **Severson et al.**, which contains cycle-level aging measurements from **124 LFP/graphite battery cells**. Each battery is tested over many charge-discharge cycles, so the dataset contains many observations per battery rather than one row per battery.

Conceptually:

```text
Battery 1
 ├── Cycle 1
 ├── Cycle 2
 ├── Cycle 3
 └── ...

Battery 2
 ├── Cycle 1
 ├── Cycle 2
 └── ...

...

Battery 124
 ├── Cycle 1
 ├── Cycle 2
 └── ...
```

This distinction is important because the project is ultimately interested in whether a model can learn degradation behaviour and generalize to battery cells that were not used for training.

For the current prototype, the reference PINN repository provides a processed `SeversonBattery.mat` file containing the arrays required for the machine-learning stage. The repository's Python implementation reads `Features_mov_Flt`, `RUL_Flt`, `PCL_Flt`, `Cycles_Flt`, `Num_Cycles_Flt`, and battery-level index arrays from this file.

Reference dataset paper:

**Severson et al., "Data-driven prediction of battery cycle life before capacity degradation", Nature Energy (2019).**

---

## SoH and RUL

Two quantities are important throughout the project.

### State of Health — SoH

SoH describes how much useful capacity a battery retains compared with its nominal capacity.

Conceptually:

```text
SoH = Remaining Capacity / Nominal Capacity
```

A healthy battery therefore has a value close to `1.0`, while a degraded battery has a lower value.

The reference preprocessing code uses a nominal capacity of `1.1 Ah`:

```text
SoH = Discharge Capacity / 1.1
```

It also defines capacity loss as:

```text
PCL = 1 - SoH
```

Therefore, for the processed dataset:

```python
soh = 1 - pcl
```

The reference implementation calculates these quantities for every cycle of every battery.

### Remaining Useful Life — RUL

RUL represents the number of cycles remaining before the battery reaches its measured end of life.

The reference preprocessing code uses:

```text
RUL = Cycle Life - Current Cycle
```

Therefore:

```text
Early in battery life → high RUL
Near end of battery life → low RUL
```

RUL will be used as a second prediction target later in the project.

---

## Feature engineering

Raw battery measurements contain a large amount of information. Instead of directly feeding every raw measurement into the neural network, the project uses a smaller set of degradation-sensitive features.

The current feature set contains **8 engineered features**:

| Feature             | What it represents                                           |
| ------------------- | ------------------------------------------------------------ |
| Slope               | Characteristic of the fitted degradation relationship        |
| Intercept           | Companion parameter from the fitted degradation relationship |
| Average Temperature | Thermal operating behaviour of the battery                   |
| Internal Resistance | Electrical condition/degradation indicator                   |
| Charge Time         | Changes in charging behaviour                                |
| dQ/dV max           | Maximum differential-capacity characteristic                 |
| dQ/dV min           | Minimum differential-capacity characteristic                 |
| dQ/dV variance      | Variation in the differential-capacity signal                |

These eight features correspond to the feature extraction used in the reference implementation. The repository's `ExtractFeature.m` calculates the slope/intercept parameters, temperature, internal resistance, charge time, and three differential-capacity statistics.

The purpose of feature engineering is to convert complex battery measurements into a compact numerical representation that can be used by a machine-learning model.

In simplified form:

```text
Raw battery measurements
          ↓
Feature extraction
          ↓
8 degradation-related features
```

---

## Data preprocessing

Before feature extraction, the reference processing pipeline performs several data-cleaning operations.

At a high level, it:

* removes empty initial cycles when present;
* removes cycles after the measured cycle life;
* identifies extreme discharge-capacity outliers;
* replaces detected outliers using neighbouring values;
* calculates SoH, PCL and RUL;
* extracts the eight features;
* applies moving-average smoothing to the feature sequences.

The processed features and targets are then flattened and stored in `SeversonBattery.mat`.

The important processed arrays are:

```text
Features_mov_Flt
RUL_Flt
PCL_Flt
Cycles_Flt
Num_Cycles_Flt
train_ind
test_ind
secondary_test_ind
```

`Num_Cycles_Flt` records how many cycles belong to each battery, which allows the flattened dataset to be reconstructed into individual battery histories.

For the current prototype, the processed `.mat` representation is used so that the project can focus on the machine-learning stage without repeatedly processing the entire raw experimental dataset.

---

## Preliminary degradation analysis

Before training any model, it is useful to verify that the processed data actually shows sensible battery degradation behaviour.

Two initial plots are used.

### SoH vs Cycle Number

```text
X-axis → Cycle Number
Y-axis → State of Health (SoH)
```

The expected behaviour is a general decrease in SoH as the number of charge-discharge cycles increases.

The graph therefore represents the **observed degradation trajectory of the battery**.

It is not a model prediction.

A suitable explanation during a project review is:

> "This graph shows the degradation trajectory of the battery over repeated charge-discharge cycles. The decrease in State of Health represents the loss of usable battery capacity as cycling progresses. We used this preliminary analysis to verify that the processed data preserves the expected battery-aging behaviour."

### RUL vs Cycle Number

```text
X-axis → Cycle Number
Y-axis → Remaining Useful Life (RUL)
```

RUL is expected to decrease as the battery approaches the end of its useful cycle life.

These plots are useful as a simple sanity check before machine-learning experiments.

```text
Processed Data
      ↓
SoH / RUL calculation
      ↓
Degradation plots
      ↓
Verify expected behaviour
      ↓
Machine-learning stage
```

---

## Preparing the data for the baseline model

Machine-learning models are commonly represented using two main variables:

```text
X = Input to the model
y = Target to predict
```

For the first baseline experiment, the model input consists of the 8 engineered features plus cycle number.

```python
X = np.hstack((features, cycles[:, None]))
```

Therefore each input row contains:

```text
1. Slope
2. Intercept
3. Average Temperature
4. Internal Resistance
5. Charge Time
6. dQ/dV max
7. dQ/dV min
8. dQ/dV variance
9. Cycle number
```

The first target is SoH:

```python
y = soh
```

So the first machine-learning problem is:

```text
8 engineered features + cycle number
                ↓
        Baseline Neural Network
                ↓
               SoH
```

RUL will be introduced as another prediction task later.

The reference implementation similarly combines the feature matrix with cycle number to form the model inputs.

---

## Why start with a baseline neural network?

The final objective is a Physics-Informed Neural Network, but it is important to first establish what can be achieved using the battery data alone.

The baseline model answers:

> **Can a normal data-driven neural network learn the relationship between battery degradation features and battery health?**

This provides a reference point for the later physics-informed model.

The project will therefore progress from:

```text
Data-driven learning
        ↓
Measure baseline performance
        ↓
Introduce battery degradation physics
        ↓
Physics-Informed Neural Network
        ↓
Compare both approaches
```

This makes it possible to determine whether the addition of physical constraints actually improves prediction quality, stability, or generalization.

The reference PINN repository follows this general idea by providing a data-driven neural-network approach alongside physics-informed approaches.

---

## Current status

The current prototype has reached the following stage:

```text
LAYER 1 — RAW DATA
        ↓
LAYER 2 — DATA LOADING
        ↓
LAYER 3 — DATA PROCESSING
        ↓
LAYER 4 — FEATURE EXTRACTION
        ↓
LAYER 5 — ML-READY DATASET
        ↓
LAYER 6 — BASELINE NEURAL NETWORK  ← CURRENT STAGE
        ↓
LAYER 7 — PREDICTION / EVALUATION
```

### Completed so far

* Dataset acquired and processed representation loaded.
* Battery-level data structure examined.
* Eight degradation-sensitive features identified.
* SoH and RUL targets established.
* Battery degradation behaviour visualized using SoH/RUL plots.
* Machine-learning input `X` created.
* Initial target `y = SoH` created.
* Baseline neural-network stage initiated.

### Still to be implemented

The next stages are intentionally left for subsequent development:

```text
Battery-wise train / validation / test split
             ↓
Baseline neural-network training
             ↓
SoH / RUL prediction
             ↓
MAE / RMSE / R² evaluation
             ↓
Physics-based degradation model
             ↓
Physics-Informed Neural Network
             ↓
Advanced physics-learning model
             ↓
Final model comparison
```

This staged approach allows the project to establish a working data-driven baseline before introducing the physics-informed component.

---

## Project architecture

The overall architecture is:

```text
┌─────────────────────────────┐
│ LAYER 1 — RAW BATTERY DATA  │
│ Voltage / Current / Temp /   │
│ Capacity / Cycle information │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ LAYER 2 — DATA LOADING      │
│ Load and organize data      │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ LAYER 3 — DATA PROCESSING   │
│ Cleaning / outliers /       │
│ SoH / RUL calculation       │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ LAYER 4 — FEATURE EXTRACTION│
│ 8 degradation features      │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ LAYER 5 — DATA MANAGEMENT   │
│ ML-ready input/target data  │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ LAYER 6 — NEURAL NETWORK    │
│ Baseline data-driven model  │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ LAYER 7 — RESULTS           │
│ SoH / RUL predictions       │
│ and model evaluation        │
└─────────────────────────────┘
```

The current work is positioned at the transition into **Layer 6**.

The SoH/RUL graphs created during preliminary analysis should not be confused with final model predictions. They show the behaviour already present in the dataset; neural-network predictions will be produced only after the model is trained and evaluated.

---

## Quick explanation for project review

A concise explanation of the work completed so far:

> "We are using the Severson lithium-ion battery aging dataset containing 124 LFP/graphite cells with cycle-level measurements. We use the processed battery representation to construct a machine-learning dataset and work with eight degradation-sensitive features, including temperature, internal resistance, charge time and differential-capacity characteristics. We calculate SoH and RUL and perform preliminary degradation analysis using SoH-versus-cycle and RUL-versus-cycle plots. We have now converted the processed data into the standard machine-learning format of inputs and targets and started implementing the baseline data-driven neural network. The next stage is battery-wise dataset splitting and baseline model training, after which physics-based constraints will be introduced to develop the PINN."

---

## References

1. Severson, K. A. et al. **Data-driven prediction of battery cycle life before capacity degradation.** *Nature Energy*, 2019.
   https://www.nature.com/articles/s41560-019-0356-8

2. WenPengfei0823. **PINN-Battery-Prognostics**. GitHub repository.
   https://github.com/WenPengfei0823/PINN-Battery-Prognostics

3. Reference files used for understanding the processing pipeline:

   * `ProcessData.m`
   * `ExtractFeature.m`
   * `functions.py`



After preparing the ML-ready dataset and implementing the baseline `DataDrivenNN`, the next stage is to make sure that the model is trained and evaluated correctly.

The main focus of these steps is:

```text
Battery-wise split
       ↓
Feature scaling
       ↓
PyTorch tensors
       ↓
Baseline NN training
       ↓
Validation
       ↓
Test prediction
       ↓
Performance evaluation