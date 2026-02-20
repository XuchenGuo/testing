# Multi-target support + Adaptive Error Rate

This document describes **two newer capabilities** added to the original `df-analyze` workflow:

1. **Multi-target learning support** (`--targets`): run the same analysis on *multiple* target columns in one invocation.
2. **Adaptive error rate analysis** (`--adaptive-error`, currently classification only, future implemented with the regression)  
   Estimates a per-sample expected error rate by mapping predictive confidence to validation error.  
   Instead of reporting only a global accuracy metric, this provides a reliability estimate for each individual prediction.

> This work builds on the original **df-analyze** framework developed by Derek Berger in the Medical Imaging & Bioinformatics Lab, under the supervision of Professor Dr.Jacob Levman (StFX, Nova Scotia, Canada).  

---

This README documents two features in this codebase:

1. **Multi-target runs** (multiple target columns in one invocation)
2. **Adaptive Error Rate (aER)** (classification-only, per-sample expected error estimates)

For the full CLI, run:

```bash
python df-analyze.py --help
```

---

## Quickstart

All examples assume you run from the repository root.

### Single-target classification

```bash
python df-analyze.py   --df path/to/data.csv   --mode classify   --target target_column
```

### Single-target classification + aER

```bash
python df-analyze.py   --df path/to/data.csv   --mode classify   --target target_column   --adaptive-error
```

### Multi-target (classification or regression)

Targets are **comma-separated**:

```bash
python df-analyze.py   --df path/to/data.csv   --mode classify   --targets y1,y2,y3
```

or:

```bash
python df-analyze.py   --df path/to/data.csv   --mode regress   --targets y1,y2,y3
```

### Multi-target + aER

aER runs **once per target**:

```bash
python df-analyze.py   --df path/to/data.csv   --mode classify   --targets y1,y2,y3   --adaptive-error
```

---

## Multi-target runs

### What “multi-target” means in df-analyze

In a multi-target run, you provide **multiple target columns** and `df-analyze` runs one pipeline that:

1. prepares one shared feature matrix `X`
2. runs target-dependent univariate analyses **per target**
3. runs feature selection **per target**, then aggregates selections across targets into a shared feature set (per selection method)
4. tunes/evaluates models on multi-output `y` (a DataFrame with one column per target), where supported

### Enable multi-target mode

Use `--targets`:

- DO `--targets y1,y2,y3`
- DO `--targets "target A",target_B` (whitespace around commas is fine)
- DO NOT `--target y1 y2 y3` (this is treated as one target name containing spaces)

### Data rules for multi-target classification

When `--mode classify` and you provide multiple targets:

- Each target column is **label-encoded** (string labels are supported).
- Row filtering is applied to the **intersection** across all targets:
  - if a row has a missing value in *any* target, it is removed for *all* targets
  - if a row belongs to a “rare” class in *any* target, it is removed for *all* targets

A class is treated as “rare” if it has **≤ 20 samples** (`N_TARG_LEVEL_MIN = 20`) and may be removed during target encoding.  
If, after filtering, any target still has a class with fewer than 20 samples, `df-analyze` will raise an error during validation.

### Feature-selection aggregation across targets

Multi-target selection is: **select per target → aggregate across targets**.

Aggregation is controlled by:

- `--mt-agg-strategy borda` *(default)*
- `--mt-agg-strategy freq`
- `--mt-top-k K` *(optional)*

#### `borda` strategy (default)

For each selection method:

1. each target produces a ranked feature list
2. feature ranks are combined via a Borda-style point system
3. scores are weighted by **support** (how many targets selected the feature)
4. features are filtered by a minimum support threshold of **20% of targets** (rounded up)
5. The final ranked feature list is limited to `--mt-top-k` when provided.

#### `freq` strategy

- features are ranked by how many targets selected them
- ties are broken by average rank (when available)
- the same `--mt-top-k` cap apply

#### for the `--mt-top-k`

`df-analyze` chooses `K` automatically:

- median of the per-target selected feature counts
- clamped to a minimum of **5** and a maximum of **80**

### Model support in multi-target mode

Multi-target tuning/evaluation requires estimator support for multi-output `y` .

In this codebase:

- **KNN**, **CatBoost**, and **Dummy** include explicit handling for multi-output predictions/probabilities.
- Other models will implemented in the future.

### Output changes in multi-target mode

Assuming your run directory is:

```
<outdir>/<run_hash>/
```

Multi-target runs add/modify outputs in a few places.

#### Univariate outputs become per-target

- `features/associations/<target>/...`
- `features/predictions/<target>/...`

#### Results include per-target performance tables

In `results/` you will see (in addition to the usual single-target files):

- `performance_long_table_per_target.csv` : long-format metrics per target
- `final_performances_per_target.csv` : compact per-target summary
- `results_report_target_<target>.md` : one markdown report per target
- `main_metric_by_target_acc.csv` *(classification)* : wide table of accuracy by target
- `main_metric_by_target_mae.csv` *(regression)* : wide table of MAE by target

`performance_long_table.csv` and `final_performances.csv` still exist and represent **overall** scores aggregated across targets (mean of per-target scores when `y` is multi-output).

#### Label mappings

For classification, label mappings are saved in:

- `prepared/labels.parquet`

In multi-target classification this stores a column per target, mapping encoded integers back to original labels.

---

## Adaptive Error Rate (aER)

### Overview

Standard model validation reports a single dataset-level error rate (e.g., 5% error, 95% accuracy).  
This assumes uniform reliability across all samples.

In practice, model reliability varies by sample.

Adaptive Error Rate (aER) provides a per-sample estimate of expected error by relating predictive confidence to empirically observed validation error.

Formally:

aER(x) ≈ P(prediction is incorrect | confidence(x))

---

### Interpretation

- Global validation error: ~3–5% (example)
- Some samples: expected error >20–30%

Thus, although overall accuracy may be high, individual predictions can carry substantially elevated risk.

aER converts model-specific confidence values into an interpretable quantity

This enables per-sample reliability assessment rather than a single global error value.

### When aER runs

aER is implemented for **classification only**.

It runs when:

- `--mode classify`
- `--adaptive-error` is enabled
- there are usable tuned model results

Notes:

- **Dummy** is skipped in aER analysis.
- By default, aER analyzes all available model families (one best result per family). Use `--aer-top-k` to limit runtime.

### Some Extra Pipeline

For each analyzed model:

1. refit tuned hyperparameters and build **out-of-fold (OOF)** predictions on the training split (`--aer-oof-folds`)
2. optionally select and apply probability calibration (auto-selected)
3. compute candidate confidence metrics (probability margins, model-specific confidence metric)
4. select the best confidence metric (unless `--aer-confidence-metric` is set)
5. fit a confidence → expected error lookup (binning + shrinkage; optional smoothing/monotonic constraints)
6. apply the mapping to the holdout test set to produce per-sample aER

- coverage vs accuracy (risk–coverage curve)
- a conservative risk-controlled threshold (controlled by `--aer-target-error`, `--aer-alpha`, `--aer-nmin`)

### Key aER flags

Enable and scope:

- `--adaptive-error` : enable aER
- `--aer-top-k K` : analyze only the top K model families (`0` = all)

OOF + mapping:

- `--aer-oof-folds` (default `5`)
- `--aer-bins` (default `20`)
- `--aer-min-bin-count` (default `10`)
- `--aer-prior-strength` (default `2.0`)
- `--no-aer-smooth` : disable smoothing (smoothing is on by default)
- `--aer-monotonic` : enforce monotonic decrease of error with confidence
- `--aer-adaptive-binning` : use quantile bins for skewed confidence distributions
- `--aer-confidence-metric <name|auto>` (default `auto`)

Risk control (best model):

- `--aer-target-error` (default `0.05`)
- `--aer-alpha` (default `0.05`)
- `--aer-nmin` (default `1`)

Ensembles (optional):

- `--aer-ensemble`
- `--aer-ensemble-strategies ...`

### aER output layout

All aER outputs live under:

```
<outdir>/<run_hash>/results/adaptive_error/
```

If you provided external test sets (`--df-train` + `--df-tests`), each test split gets a subfolder:

```
.../results/adaptive_error/test00/
.../results/adaptive_error/test01/
...
```

Typical contents:

```
results/adaptive_error/
  run_config.json
  plots/
    confidence_vs_expected_error_compare.png
  tables/
    models_ranked.csv
    aer_metrics_by_model.csv
  predictions/
    test_per_sample_multi_model.csv
    test_per_sample_multi_model.parquet
  models/
    <model_slug>/
      metadata/
      tables/
      plots/
      predictions/
      reports/
  ensemble/                      # only if --aer-ensemble
    ...
```

Best-model extras (inside that model’s folder):

- `tables/clinician_view.csv` and `reports/clinician_view.md`
- `tables/top20_highest_adaptive_error.csv`
- `tables/coverage_accuracy_curve.csv`, `tables/coverage_summary.csv`, and `plots/coverage_vs_accuracy.png`
- `metadata/risk_control_threshold.json` and `reports/risk_control_threshold.md`

#### `--no-preds` and per-sample outputs

`--no-preds` disables per-sample prediction outputs. aER will still run, but per-sample CSV/Parquet files are replaced by small placeholder files.

---

## Multi-target + aER together

If you run multi-target mode with `--adaptive-error`:

- aER is run **separately for each target**
- outputs are organized as:

```
results/adaptive_error/<target>/
```

or, with external test sets:

```
results/adaptive_error/test00/<target>/
results/adaptive_error/test01/<target>/
...
```

---

## Common running

- `--targets` is comma-separated: `--targets y1,y2,y3`
- `--classifiers` / `--regressors` are **space-separated** (no commas), e.g.:

  ```bash
  --classifiers dummy knn lgbm
  ```

- aER is classification-only (`--mode classify`)
- multi-target classification may drop many rows because filtering is applied across all targets

---

## Reproducibility

Each run writes `options.json` in the run directory. When aER is enabled, `results/adaptive_error/run_config.json` also records the aER configuration and which models were analyzed.
