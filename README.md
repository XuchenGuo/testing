# df-analyze User Guide — Multi-Target Support and Adaptive Error Rate

Package version: 4.1.0 
Python requirement >= 3.13.11

Main features covered: `--targets` and `--adaptive-error`

This guide covers the theory, command-line usage, output interpretation, and
installation details for two newly integrated features: multi-target support
and adaptive error rate.

# GPU Setup and Verification

The current CLI does not expose a single global `--gpu` switch. Instead, GPU
usage is decided inside model code. The `CatBoost` model sets
`task_type='GPU'` automatically when a supported GPU is detected, and the
`Gandalf` model checks `torch.cuda.is_available()` to decide whether to use a
GPU accelerator. PyTorch, NVIDIA, and CatBoost all provide official
documentation for the relevant compatibility and verification steps:

- [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [PyTorch `torch.cuda.is_available`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html)
- [PyTorch Get Started Locally](https://pytorch.org/get-started/locally/)

| Step | Command | Expected output |
|------|---------|-----------------|
| Check that the NVIDIA driver is visible | `nvidia-smi` | GPU name, driver version, memory information, and runtime status |
| Check PyTorch CUDA visibility | `python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"` | `True` if PyTorch can use CUDA; the second value is the CUDA runtime version used by that torch build |
| Check CatBoost GPU visibility | `python -c "from catboost.utils import get_gpu_device_count; print(get_gpu_device_count())"` | A positive number means CatBoost sees one or more GPUs |
| Install a matching PyTorch build | Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) | Generates the install command matching your OS, package manager, and CUDA runtime |
| Confirm driver and CUDA compatibility | Read [NVIDIA CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/) | If the driver is too old for the CUDA family used by your PyTorch build, CUDA may not initialize correctly |

All three checks in sequence:

```shell
nvidia-smi

python -c "import torch; print('torch', torch.__version__); \
    print('cuda runtime', torch.version.cuda); \
    print('cuda available', torch.cuda.is_available())"

python -c "from catboost.utils import get_gpu_device_count; \
    print('catboost GPU count', get_gpu_device_count())"
```



# Multi-Target Support

Multi-target support means that one input row is used to predict more than one
output column simultaneously. If the targets are continuous the task is
multi-output regression; if they are categorical the task is multi-output or
multiclass-multioutput classification.

The model learns a function from a feature matrix **X** to a target vector
**y** = (y₁, y₂, ...). This is valuable when the targets are related — for
example, several clinical outcomes or several laboratory measurements recorded
from the same subject.

If `--adaptive-error` is also enabled and the task is classification,
`df-analyze` slices the final multi-target results back into one target at a
time and runs adaptive error analysis separately for each target.


## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--target` | `["target"]` | Single-target mode. Use this when you have only one target column. |
| `--targets` | `[]` | Comma-separated target columns for a multi-target run. This is the key flag for this feature. |
| `--mode` | `"classify"` | Set to `classify` or `regress`. All targets in one run must use the same mode. |
| `--classifiers` | `knn dummy catboost` | Space-separated classifier list for classification runs. |
| `--regressors` | `knn dummy catboost` | Space-separated regressor list for regression runs. |
| `--mt-agg-strategy` | `"borda"` | How per-target feature selections are combined. Choices: `borda` or `freq`. |
| `--mt-top-k` | `None` | Optional final feature cap after aggregation. Omitting it triggers automatic top-k selection rather than retaining everything. |
| `--outdir` | user-defined | Where the run directory will be written. |
| `--no-preds` | flag | Skips large prediction-oriented output files to save disk space. |

Use `--targets` with a single comma-separated string, for example:

```shell
--targets outcome_1,outcome_2,outcome_3
```

If target names contain spaces, quote the entire comma-separated argument:

```shell
--targets "Outcome A,Outcome B,Outcome C"
```

Use `--classifiers` and `--regressors` as space-separated lists (not
comma-separated):

```shell
--classifiers dummy knn catboost
```


## Usage Instructions

1. Choose your task type with `--mode classify` or `--mode regress`.
2. Replace the old single-target flag `--target` with `--targets` and supply
   the target column names as one comma-separated argument.
3. Choose the model family: `--classifiers` for classification or
   `--regressors` for regression.
4. Optionally choose how `df-analyze` aggregates selected features across
   targets with `--mt-agg-strategy borda` or `--mt-agg-strategy freq`.
5. Set an output directory with `--outdir`.
6. Run the command from the repository root.


## Command Examples

Simple run (multi-target classification):

```shell
python df-analyze.py \
    --df path/to/data.csv \
    --mode classify \
    --targets outcome_a,outcome_b,outcome_c \
    --outdir ./out_mt_cls
```

More explicit run (multi-target regression):

```shell
python df-analyze.py \
    --df path/to/data.csv \
    --mode regress \
    --targets y_height,y_weight,y_age \
    --regressors dummy knn catboost \
    --mt-agg-strategy borda \
    --outdir ./out_mt_reg
```

Full-control run (multi-target classification with a fixed feature cap):

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


## Data Cleaning Rules

- Multi-target classification and regression: rows with a missing value in any
  target are always removed.
- A rare class is defined as a class with 20 or fewer samples.
- Multi-target classification: a row is removed when more than one third of its
  target values belong to rare classes.
- After filtering, the preparation code still warns if any target column retains
  classes with 20 or fewer samples, because downstream cross-validation may
  become unstable or fail.


## Multi-Target Outputs

| Path | Description |
|------|-------------|
| `prepared/y.parquet` | Prepared target data after preprocessing. For classification, these are encoded integers; for regression, these are scaled target values. |
| `prepared/labels.parquet` | For multi-target classification, stores one label map per target so encoded integers can be translated back to original class labels. |
| `features/associations/<target>/` | Per-target univariate association outputs. |
| `features/predictions/<target>/` | Per-target univariate prediction outputs when prediction outputs are enabled. |
| `results/results_report.md` | Overall report for the final multi-output evaluation. |
| `results/results_report_target_<target>.md` | One readable report per target. |
| `results/final_performances.csv` | Overall summary table. In multi-target runs this is the main aggregated performance table across targets; some models may also include joint metrics such as subset accuracy, Hamming loss, or multi-RMSE. |
| `results/final_performances_per_target.csv` | Compact per-target performance summary. |
| `results/performance_long_table_per_target.csv` | Long-form per-target metric table; usually the best file for detailed comparison. |
| `results/main_metric_by_target_acc.csv` or `main_metric_by_target_mae.csv` | One key metric per target: accuracy for classification or MAE for regression. |



# Adaptive Error Rate

Adaptive Error Rate (AER) is a sample-level reliability estimate for
classification predictions. Rather than reporting a single global error rate
for the entire dataset, AER estimates the expected probability that an
individual prediction is incorrect. This concept is closely related to
probability calibration, selective prediction, and risk-coverage control.

For a prediction on sample *x*, the adaptive error rate *aER(x)* represents
the estimated likelihood of error. A smaller *aER(x)* indicates a more
reliable prediction, while a larger value suggests lower trustworthiness. The
framework translates calibrated prediction confidence into sample-level error
rate estimates, enabling adaptive reliability assessment, improved ensemble
strategies, and more trustworthy decision-making in biomedical machine
learning.


## When AER Runs

- AER runs only for classification tasks.
- It must be explicitly enabled with `--adaptive-error`.
- If the run is regression, AER is skipped.
- If only a `Dummy` model is available after tuning, AER is skipped.
- In multi-target classification, AER runs separately for each target after the
  final multi-target evaluation is complete.


## What the AER Pipeline Does

1. Refits tuned model settings and builds out-of-fold (OOF) predictions on the
   training portion.
2. Builds or normalizes class probabilities. External calibration methods
   available include: `none`, temperature scaling, Platt scaling, isotonic, or
   one-vs-rest isotonic, chosen depending on the problem structure.
3. Constructs candidate confidence signals, which can include probability
   margin, tree vote agreement, tree leaf support, KNN vote, KNN
   distance-weighted confidence, and KNN minimum distance confidence.
4. Selects the best confidence signal automatically when
   `--aer-confidence-metric auto` is used. The selection criterion is the
   lowest cross-fitted Brier score for predicting whether the model is wrong.
5. Fits a confidence-to-expected-error mapping by binning OOF confidences,
   shrinking noisy bins toward the global error rate, smoothing the curve by
   default, and optionally enforcing monotonicity.
6. Applies the learned mapping to the holdout test set to produce one expected
   error estimate per sample.
7. Builds risk-controlled operating points using exact one-sided
   Clopper-Pearson bounds with Bonferroni adjustment over scanned thresholds.


## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--adaptive-error` | `False` | Turns AER analysis on. |
| `--aer-oof-folds` | `5` | Number of OOF splits used for AER fitting and cross-fitting. |
| `--aer-bins` | `20` | Nominal number of confidence bins. |
| `--aer-min-bin-count` | `10` | Minimum samples per bin before bins are merged or reduced. |
| `--aer-prior-strength` | `2.0` | Strength of the beta-prior shrinkage toward the global error rate. |
| `--no-aer-smooth` | False flag | Disables the default local smoothing of the error mapping. |
| `--aer-monotonic` | `False` | Enforces a monotonic confidence-to-error mapping. |
| `--aer-adaptive-binning` | `False` | Uses quantile-like adaptive bins instead of fixed-width bins. |
| `--aer-confidence-metric` | `"auto"` | Which confidence signal to use. `auto` chooses the best available signal by cross-fitted Brier score. |
| `--aer-nmin` | `1` | Minimum accepted sample count when choosing a risk-controlled threshold. |
| `--aer-target-error` | `0.05` | Target error rate for risk-controlled summaries. |
| `--aer-alpha` | `0.05` | Significance level used in the exact upper-bound calculation. |
| `--aer-top-k` | `0` | If greater than 0, run AER only on the top-k tuned base models. `0` means all usable base models. |
| `--no-preds` | False flag | Suppresses large per-sample AER output files and writes placeholders instead. |


## Confidence Metrics

| Metric | Description |
|--------|-------------|
| `proba_margin` | Confidence from the margin between top class probabilities. |
| `tree_vote_agreement` | Confidence from agreement among tree votes. |
| `tree_leaf_support` | Confidence from leaf support in tree structures. |
| `knn_vote` | Confidence from nearest-neighbor vote agreement. |
| `knn_dist_weighted` | Confidence from distance-weighted KNN behavior. |
| `knn_min_dist` | Confidence derived from nearest-neighbor distance. |
| `auto` | Let the code choose the best available signal by cross-fitted Brier score. |


## Usage Instructions

1. Make sure the task is classification: use `--mode classify`.
2. Add the flag `--adaptive-error`.
3. Optionally limit the number of analyzed models with `--aer-top-k` if disk
   space or runtime is a concern.
4. Optionally enable `--aer-adaptive-binning` for a more balanced binning
   strategy when confidence values are heavily skewed.
5. Optionally set a deployment target with `--aer-target-error`, for example
   `0.05` for a 5% error ceiling.
6. Run the command and open the `adaptive_error` folder inside the run
   directory.


## Command Examples

Simple run (single-target classification with AER):

```shell
python df-analyze.py \
    --df path/to/data.csv \
    --mode classify \
    --target target_column \
    --adaptive-error \
    --outdir ./out_aer_simple
```

More explicit run (top models only, adaptive bins, 5% risk target):

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

Full-control run (manual AER settings):

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

Combined run (multi-target classification plus AER per target):

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


## Interpreting AER Outputs

The base AER directory depends on whether the run is single-target or
multi-target:

```
# Single-target classification:
<outdir>/results/adaptive_error/

# Multi-target classification:
<outdir>/results/adaptive_error/<target_name>/
```


### Cross-Model Files

| File | Description |
|------|-------------|
| `run_config.json` | Records the AER configuration, chosen models, and run metadata. Useful for reproducibility. |
| `tables/models_ranked.csv` | Ranks the analyzed base models. Shows which models were included and where each model's folder lives. |
| `tables/aer_metrics_by_model.csv` | Cross-model error-quality summary. Focus on `global_error_test`, `brier_error_test`, and `ece_error_test`. |
| `plots/confidence_vs_expected_error_compare.png` | Visual comparison of confidence-to-error behavior across analyzed models. |
| `predictions/test_per_sample_multi_model.csv` | One test row with multiple model-specific AER columns. Useful for comparing models on the same samples. |


### Per-Model AER Folder

| File | Description |
|------|-------------|
| `metadata/proba_calibrator.json` | Which external probability calibration method was used. |
| `metadata/confidence_metric_selection.json` | Which confidence signal was selected, and the Brier-score comparison among candidate signals. |
| `metadata/adaptive_error_metrics.json` | Global error summary for the test set. Key fields: `global_error_test`, `brier_error_test`, `ece_error_test`. |
| `tables/oof_confidence_error_bins.csv` | OOF bin summary used to fit the confidence-to-error mapping. |
| `tables/test_confidence_error_bins.csv` | How the mapping behaves on the test set. |
| `tables/test_error_reliability_bins.csv` | Reliability table used for calibration-style error analysis. |
| `tables/coverage_accuracy_curve.csv` | Coverage vs. selective accuracy as samples are accepted in order of lower predicted risk. |
| `tables/coverage_summary.csv` | Short summary of a few operating points from the full coverage curve. |
| `tables/clinician_view.csv` | Practical table with row ID, true label, predicted label, `aer_pct`, and whether the row exceeds the target error. |
| `predictions/oof_per_sample.csv` | Row-level OOF diagnostic file. Good for checking how the mapping was learned. |
| `predictions/test_per_sample.csv` | Row-level test file. Main per-sample output for end users. |
| `reports/clinician_view.md` | Simplified markdown report for non-technical readers. |


### Columns in `test_per_sample.csv`

| Column | Description |
|--------|-------------|
| `row_id` | Original row index carried into the output. |
| `y_true` / `y_pred` | Encoded true and predicted class IDs. |
| `y_true_label` / `y_pred_label` | Decoded class labels when a label map is available. |
| `correct` | `1` if the prediction is correct, else `0`. |
| `confidence` | The selected confidence signal after any transformation. |
| `aer` | Estimated sample-level error probability. |
| `aer_pct` | The same estimated error expressed as a percentage. |
| `flag_gt_target_error` | `1` if `aer` is greater than or equal to the chosen `--aer-target-error`. |
| `p_max`, `p_2nd`, `p_margin` | Probability diagnostics from calibrated class probabilities. |
| `p_pred`, `p_pred_margin` | Probability diagnostics tied to the predicted class. |


### Key Metrics

| Metric | Description |
|--------|-------------|
| `global_error_test` | The ordinary test-set error rate of the classifier. This is the baseline before any selective filtering. |
| `brier_error_test` | Mean squared error between the predicted sample-level error probabilities and the actual correct/incorrect outcomes. Smaller is better. |
| `ece_error_test` | Calibration-style expected calibration error for the predicted error probabilities. Smaller is better. |


## Troubleshooting

- If no `adaptive_error` folder appears, check whether the run accidentally
  used `--mode regress` instead of `--mode classify`.
- If no AER folders appear at all, check whether only `Dummy` in the
  earlier tuning stages; AER is skipped in that case.
- If per-sample outputs are missing, check whether `--no-preds` was passed. The
  code intentionally writes placeholder files instead of full CSV or Parquet
  outputs when that flag is set.
- If the confidence vs. error plot looks noisy, try enabling
  `--aer-adaptive-binning` or increasing `--aer-min-bin-count` slightly.
