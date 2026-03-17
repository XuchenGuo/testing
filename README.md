<!-- omit from toc -->
# Contents

- [Overview](#overview)
  - [Multi-Target Support](#multi-target-support)
  - [Adaptive Error Rate](#adaptive-error-rate)
- [Installation](#installation)
  - [Python Version](#python-version)
  - [GPU Setup and Verification](#gpu-setup-and-verification)
- [Usage](#usage)
  - [Using Multi-Target Support](#using-multi-target-support)
    - [Multi-Target Command Line Options](#multi-target-command-line-options)
    - [Multi-Target Quick Start and Examples](#multi-target-quick-start-and-examples)
    - [Data Preparation Rules](#data-preparation-rules)
  - [Using Adaptive Error Rate](#using-adaptive-error-rate)
    - [When AER Runs](#when-aer-runs)
    - [AER Command Line Options](#aer-command-line-options)
    - [Confidence Metrics](#confidence-metrics)
    - [AER Quick Start and Examples](#aer-quick-start-and-examples)
- [Analysis Pipeline](#analysis-pipeline)
  - [Multi-Target Analysis](#multi-target-analysis)
  - [Adaptive Error Rate Pipeline](#adaptive-error-rate-pipeline)
- [Program Outputs](#program-outputs)
  - [Multi-Target Outputs](#multi-target-outputs)
  - [Adaptive Error Rate Outputs](#adaptive-error-rate-outputs)
    - [Cross-Model Files](#cross-model-files)
    - [Model-Specific Files](#model-specific-files)
    - [Key Columns and Metrics](#key-columns-and-metrics)
- [Limitations](#limitations)
  - [Multi-Target Constraints](#multi-target-constraints)
  - [Adaptive Error Rate Constraints](#adaptive-error-rate-constraints)
- [Currently Implemented Program Features and Analyses](#currently-implemented-program-features-and-analyses)
  - [Completed Features](#completed-features)
    - [Multi-Target Prediction](#multi-target-prediction)
    - [Adaptive Error Estimation](#adaptive-error-estimation)

# Overview

This update documents the additions introduced in `df-analyze` 4.1.0:
multi-target support through `--targets`, and Adaptive Error Rate (AER)
analysis for classification through `--adaptive-error`.

The guide below follows the section organization and tone of the main
repository README, but focuses only on material that is new.

## Multi-Target Support

Multi-target support allows one input row to predict more than one output
column in the same run. For continuous targets, this is multi-output
regression. For categorical targets, this is multi-output classification.

This is most useful when the targets are related, for example several
clinical outcomes or several laboratory values. The model learns a mapping
from a feature matrix `X` to a target vector `(y1, y2, ...)` rather than to a
single target column.

If `--adaptive-error` is also enabled for a classification run,
`df-analyze` completes the final multi-target evaluation first, then slices
the results back into one target at a time and runs AER separately for each
target.

## Adaptive Error Rate

Adaptive Error Rate is a sample-level reliability estimate for classification
predictions. Instead of reporting only one global error rate for the full test
set, AER estimates the expected probability that an individual prediction is
incorrect.

Lower AER values indicate more reliable predictions. Higher values indicate
less trustworthy predictions. In practice, this places AER close to
probability calibration, selective prediction, and risk-coverage analysis:
the goal is not only to predict well on average, but also to estimate how
much trust to place in each prediction.

# Installation

The recommended installation path remains the `uv` workflow described in the
main README. The additions documented here assume Python 3.13.11 or newer.
Examples below assume you are running commands from the repository root.

## Python Version

```shell
uv python install 3.13.11
uv sync
uv run python df-analyze.py --help
```

## GPU Setup and Verification

The current CLI does not expose one global `--gpu` switch. GPU use is decided
inside model code. At present, CatBoost enables `task_type='GPU'` when a
supported device is detected, and Gandalf checks
`torch.cuda.is_available()` before using a GPU accelerator.

For compatibility and installation details, see the
[NVIDIA CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/),
[PyTorch CUDA availability reference](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html),
and the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

A minimal verification sequence is:

1. Confirm that the NVIDIA driver is visible with `nvidia-smi`.
2. Confirm that PyTorch can see CUDA.
3. Confirm that CatBoost can see one or more GPUs.
4. Make sure the installed PyTorch build matches the CUDA runtime supported
   by the driver.

```shell
nvidia-smi

python -c "import torch; print('torch', torch.__version__); print('cuda runtime', torch.version.cuda); print('cuda available', torch.cuda.is_available())"

python -c "from catboost.utils import get_gpu_device_count; print('catboost GPU count', get_gpu_device_count())"
```

# Usage

As in the main README, the examples below omit `uv run` for brevity.

## Using Multi-Target Support

Use `--targets` when one run should predict more than one target column.
Pass the target names as a single comma-separated argument. If a target name
contains spaces, quote the full argument, for example:

```shell
--targets "Outcome A,Outcome B,Outcome C"
```

Model lists remain space-separated, not comma-separated. For example:

```shell
--classifiers dummy knn catboost
--regressors dummy knn catboost
```

### Multi-Target Command Line Options

- `--target`
  - Single-target mode. Use this when there is only one target column.
- `--targets`
  - Comma-separated target columns for a multi-target run.
- `--mode`
  - Set to `classify` or `regress`. All targets in one run must match the
    same mode.
- `--classifiers`
  - Space-separated classifier list for multi-target classification runs.
- `--regressors`
  - Space-separated regressor list for multi-target regression runs.
- `--mt-agg-strategy`
  - Combine per-target feature selections by `borda` or `freq`.
- `--mt-top-k`
  - Optional final feature cap after aggregation. In the current
    implementation, omitting this option triggers automatic top-k selection
    rather than keeping all aggregated features.
- `--outdir`
  - Directory where the run outputs will be written.
- `--no-preds`
  - Skip large prediction-oriented output files to reduce disk use.

### Multi-Target Quick Start and Examples

Simple multi-target classification run:

```shell
python df-analyze.py \
    --df path/to/data.csv \
    --mode classify \
    --targets outcome_a,outcome_b,outcome_c \
    --outdir ./out_mt_cls
```

More explicit multi-target regression run:

```shell
python df-analyze.py \
    --df path/to/data.csv \
    --mode regress \
    --targets y_height,y_weight,y_age \
    --regressors dummy knn catboost \
    --mt-agg-strategy borda \
    --outdir ./out_mt_reg
```

Multi-target classification with a fixed feature cap:

```shell
python df-analyze.py \
    --df path/to/data.csv \
    --mode classify \
    --targets diagnosis_1,diagnosis_2,diagnosis_3 \
    --classifiers dummy knn catboost \
    --mt-agg-strategy freq \
    --mt-top-k 40 \
    --test-val-size 0.20 \
    --htune-trials 50 \
    --seed 69 \
    --outdir ./out_mt_full
```

### Data Preparation Rules

For multi-target runs, target preprocessing is stricter than in the
single-target case because every row contributes to several outcomes at once.

- Multi-target classification:
  - rows with a missing value in any target are removed
  - a rare class is currently defined as a class with 20 or fewer samples
  - a row is removed when more than one third of its target values belong to
    rare classes
  - after filtering, `df-analyze` still warns if any target column retains a
    class with 20 or fewer samples, since downstream cross-validation may
    become unstable or fail
- Multi-target regression:
  - rows with a missing value in any target are removed

## Using Adaptive Error Rate

Use `--adaptive-error` to estimate a per-sample error probability for
classification predictions. The output is a risk score attached to each
prediction, not a replacement for the ordinary model metrics already reported
elsewhere in `df-analyze`.

### When AER Runs

AER currently runs only for classification tasks.

- You must enable it with `--adaptive-error`.
- If the run is regression, AER is skipped.
- If only a Dummy model remains after tuning, AER is skipped.
- In multi-target classification, AER runs separately for each target after
  the final multi-target evaluation completes.

### AER Command Line Options

- **Core switch**
  - `--adaptive-error`
    - Enable AER analysis.
- **Cross-fitting and binning**
  - `--aer-oof-folds` (default `5`)
    - Number of out-of-fold splits used for AER fitting and related
      cross-fitting.
  - `--aer-bins` (default `20`)
    - Nominal number of confidence bins.
  - `--aer-min-bin-count` (default `10`)
    - Minimum number of samples per bin before bins are merged or reduced.
  - `--aer-prior-strength` (default `2.0`)
    - Strength of the beta-prior shrinkage toward the global error rate.
  - `--aer-adaptive-binning`
    - Use quantile-like adaptive bins instead of fixed-width bins.
  - `--no-aer-smooth`
    - Disable the default local smoothing of the confidence-to-error mapping.
  - `--aer-monotonic`
    - Enforce a monotonic confidence-to-error mapping.
- **Confidence signal**
  - `--aer-confidence-metric` (default `auto`)
    - Confidence signal used to build the mapping. With `auto`,
      `df-analyze` selects the best available signal by cross-fitted Brier
      score.
- **Risk-controlled summaries**
  - `--aer-target-error` (default `0.05`)
    - Target error rate used for summary tables and thresholding.
  - `--aer-alpha` (default `0.05`)
    - Significance level for the exact upper-bound calculation.
  - `--aer-nmin` (default `1`)
    - Minimum accepted sample count when choosing a risk-controlled
      threshold.
- **Model selection and outputs**
  - `--aer-top-k` (default `0`)
    - If greater than `0`, run AER only on the top-k tuned base models.
      `0` means all usable base models.
  - `--no-preds`
    - Suppress large per-sample AER output files and write placeholders
      instead.

### Confidence Metrics

The current implementation can score confidence using several model-dependent
signals:

- `proba_margin`
  - Margin between the top two class probabilities.
- `tree_vote_agreement`
  - Agreement among tree votes.
- `tree_leaf_support`
  - Leaf support or training support within a tree model.
- `knn_vote`
  - Nearest-neighbor vote agreement.
- `knn_dist_weighted`
  - Distance-weighted nearest-neighbor confidence.
- `knn_min_dist`
  - Confidence derived from nearest-neighbor distance.
- `auto`
  - Select the best available signal by cross-fitted Brier score.

### AER Quick Start and Examples

Simple single-target classification run with AER:

```shell
python df-analyze.py \
    --df path/to/data.csv \
    --mode classify \
    --target target_column \
    --adaptive-error \
    --outdir ./out_aer_simple
```

More explicit AER run with adaptive binning and a 5% target error rate:

```shell
python df-analyze.py \
    --df path/to/data.csv \
    --mode classify \
    --target target_column \
    --classifiers dummy knn lgbm catboost lr \
    --adaptive-error \
    --aer-top-k 3 \
    --aer-adaptive-binning \
    --aer-target-error 0.05 \
    --outdir ./out_aer_mid
```

AER run with manual settings:

```shell
python df-analyze.py \
    --df path/to/data.csv \
    --mode classify \
    --target target_column \
    --classifiers dummy knn lgbm catboost lr sgd \
    --adaptive-error \
    --aer-oof-folds 5 \
    --aer-bins 20 \
    --aer-min-bin-count 10 \
    --aer-prior-strength 2.0 \
    --aer-monotonic \
    --aer-confidence-metric auto \
    --aer-target-error 0.05 \
    --aer-alpha 0.05 \
    --aer-nmin 25 \
    --outdir ./out_aer_full
```

Multi-target classification with per-target AER:

```shell
python df-analyze.py \
    --df path/to/data.csv \
    --mode classify \
    --targets y1,y2,y3 \
    --classifiers dummy knn lgbm catboost \
    --mt-agg-strategy borda \
    --mt-top-k 40 \
    --adaptive-error \
    --aer-adaptive-binning \
    --aer-top-k 3 \
    --aer-target-error 0.05 \
    --outdir ./out_mt_aer
```

# Analysis Pipeline

## Multi-Target Analysis

Relative to the single-target workflow, the multi-target path adds one key
step: per-target results must be combined before final model tuning and
evaluation.

The process is:

1. load and preprocess the dataset, removing rows that violate the
   multi-target target-cleaning rules
2. compute per-target univariate association outputs and, when enabled,
   per-target univariate prediction outputs
3. perform per-target feature selection
4. aggregate selected features across targets with `--mt-agg-strategy`
   (`borda` or `freq`)
5. optionally cap the aggregated set with `--mt-top-k`
6. tune and evaluate multi-output models on the final feature set
7. write overall and per-target performance reports
8. if `--adaptive-error` is enabled for classification, run AER separately
   for each target after the final multi-target evaluation

## Adaptive Error Rate Pipeline

AER is fit after the main classification model has already been selected and
tuned. The current AER workflow is:

1. refit the tuned model settings and build out-of-fold predictions on the
   training portion
2. build or normalize class probabilities, with optional external
   calibration methods such as none, temperature scaling, Platt scaling,
   isotonic, or one-vs-rest isotonic depending on the problem structure
3. construct candidate confidence signals, including probability margin,
   tree vote agreement, tree leaf support, nearest-neighbor vote agreement,
   distance-weighted nearest-neighbor confidence, and nearest-neighbor
   distance confidence
4. if `--aer-confidence-metric auto` is used, choose the best available
   confidence signal by the lowest cross-fitted Brier score for predicting
   whether the model is wrong
5. fit a confidence-to-expected-error mapping by binning out-of-fold
   confidences, shrinking noisy bins toward the global error rate, smoothing
   the curve by default, and optionally enforcing monotonicity
6. apply the learned mapping to the holdout test set to produce one expected
   error estimate per sample
7. build risk-controlled operating points using exact one-sided
   Clopper-Pearson style bounds with Bonferroni adjustment over scanned
   thresholds

# Program Outputs

## Multi-Target Outputs

The multi-target workflow adds the following outputs beyond the standard
single-target directories:

```text
prepared/
├── labels.parquet
└── y.parquet

features/
├── associations/<target>/
└── predictions/<target>/

results/
├── final_performances.csv
├── final_performances_per_target.csv
├── main_metric_by_target_<metric>.csv
├── performance_long_table_per_target.csv
├── results_report.md
└── results_report_target_<target>.md
```

- `prepared/y.parquet`
  - Prepared target data after preprocessing. For classification, these are
    encoded integers. For regression, these are scaled target values.
- `prepared/labels.parquet`
  - For multi-target classification, this stores one label map per target so
    encoded integers can be translated back to original class labels.
- `features/associations/<target>/`
  - Per-target univariate association outputs.
- `features/predictions/<target>/`
  - Per-target univariate prediction outputs when prediction outputs are
    enabled.
- `results/results_report.md`
  - Overall report for the final multi-output evaluation.
- `results/results_report_target_<target>.md`
  - Readable report for one target.
- `results/final_performances.csv`
  - Main aggregated performance table across targets. Depending on model and
    task, this may also include joint metrics such as subset accuracy,
    hamming loss, or multi-RMSE.
- `results/final_performances_per_target.csv`
  - Compact per-target performance summary.
- `results/performance_long_table_per_target.csv`
  - Long-form per-target metric table, usually the best file for detailed
    comparison.
- `results/main_metric_by_target_<metric>.csv`
  - One key metric per target: `acc` for classification or `mae` for
    regression.

## Adaptive Error Rate Outputs

The base AER directory is:

```text
Single-target classification:
<outdir>/results/adaptive_error/

Multi-target classification:
<outdir>/results/adaptive_error/<target_name>/
```

### Cross-Model Files

These files summarize all analyzed AER models for a given target.

- `run_config.json`
  - Records the AER configuration, chosen models, and run metadata.
- `tables/models_ranked.csv`
  - Ranks the analyzed base models and shows where each model's folder lives.
- `tables/aer_metrics_by_model.csv`
  - Cross-model error quality summary. Focus on `global_error_test`,
    `brier_error_test`, and `ece_error_test`.
- `plots/confidence_vs_expected_error_compare.png`
  - Visual comparison of confidence-to-error behavior across analyzed models.
- `predictions/test_per_sample_multi_model.csv`
  - One test row with multiple model-specific AER columns for side-by-side
    comparison.

### Model-Specific Files

Within one model's AER folder, the most important outputs are:

- `metadata/proba_calibrator.json`
  - External probability calibration method used for the model.
- `metadata/confidence_metric_selection.json`
  - Winning confidence signal, with the Brier-score comparison among
    candidate signals.
- `metadata/adaptive_error_metrics.json`
  - Global error summary for the test set. The key fields are
    `global_error_test`, `brier_error_test`, and `ece_error_test`.
- `tables/oof_confidence_error_bins.csv`
  - Out-of-fold bin summary used to fit the mapping.
- `tables/test_confidence_error_bins.csv`
  - How the learned mapping behaves on the test set.
- `tables/test_error_reliability_bins.csv`
  - Reliability table used for calibration-style error analysis.
- `tables/coverage_accuracy_curve.csv`
  - Coverage versus selective accuracy as samples are accepted in order of
    lower predicted risk.
- `tables/coverage_summary.csv`
  - Short summary of operating points from the full coverage curve.
- `tables/clinician_view.csv`
  - Practical summary with `row_id`, true label, predicted label, `aer_pct`,
    and whether the row exceeds the target error.
- `predictions/oof_per_sample.csv`
  - Row-level out-of-fold diagnostic file used to inspect how the mapping was
    learned.
- `predictions/test_per_sample.csv`
  - Main row-level AER output for the holdout test set.
- `reports/clinician_view.md`
  - Simplified markdown report for non-technical readers.

### Key Columns and Metrics

The main columns in `predictions/test_per_sample.csv` are:

- `row_id`
  - Original row index carried into the output.
- `y_true` / `y_pred`
  - Encoded true and predicted class IDs.
- `y_true_label` / `y_pred_label`
  - Decoded class labels when a label map is available.
- `correct`
  - `1` if the prediction is correct, otherwise `0`.
- `confidence`
  - Selected confidence signal after any transformation.
- `aer`
  - Estimated sample-level error probability.
- `aer_pct`
  - The same estimated error, shown as a percentage.
- `flag_gt_target_error`
  - `1` if `aer` is greater than or equal to the chosen
    `--aer-target-error`.
- `p_max`, `p_2nd`, `p_margin`
  - Probability diagnostics from calibrated class probabilities.
- `p_pred`, `p_pred_margin`
  - Probability diagnostics tied to the predicted class.

The main summary metrics are:

- `global_error_test`
  - Ordinary test-set error rate of the classifier.
- `brier_error_test`
  - Mean squared error between predicted sample-level error probabilities and
    actual incorrect/correct outcomes. Smaller is better.
- `ece_error_test`
  - Calibration-style expected calibration error for the predicted error
    probabilities. Smaller is better.

# Limitations

## Multi-Target Constraints

- All targets in one run must share the same `--mode`.
- Rows with missing target values are removed before analysis.
- Rare classes can trigger additional row removal in multi-target
  classification, and remaining rare classes may still make downstream
  cross-validation unstable.
- Feature aggregation is currently limited to `borda` and `freq`.
- Omitting `--mt-top-k` does not preserve all aggregated features. In the
  current implementation it triggers automatic top-k selection.

## Adaptive Error Rate Constraints

- AER currently runs only for classification tasks.
- AER is skipped if only a Dummy model survives tuning.
- If `--no-preds` is used, large per-sample AER files are intentionally
  suppressed.
- If no AER folder appears, first confirm that the run was classification and
  that at least one non-Dummy model survived tuning.
- If the confidence-to-error plot is noisy, try
  `--aer-adaptive-binning` or a slightly larger `--aer-min-bin-count`.

# Currently Implemented Program Features and Analyses

## Completed Features

### Multi-Target Prediction

- Multi-output classification and regression through `--targets`
- Per-target feature selection combined by `borda` or `freq`
- Overall and per-target performance reports
- Optional final feature cap through `--mt-top-k`
- Per-target association and prediction directories

### Adaptive Error Estimation

- Sample-level expected error estimates for classification predictions
- Cross-fitted confidence signal selection through `--aer-confidence-metric auto`
- Confidence-to-error mapping with shrinkage, smoothing, and optional
  monotonicity
- Risk-controlled coverage summaries based on a target error level
- Per-sample CSV outputs and a clinician-facing markdown report
