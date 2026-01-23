# df-analyze — Adaptive Error Rate (aER) & Multi-Target Learning Integration  

Based on the original **df-analyze** framework developed by Derek Berger in the Medical Imaging & Bioinformatics Lab, under the supervision of Jacob Levman at St. Francis Xavier University (StFX), Nova Scotia, Canada.  
Upstream repository: https://github.com/stfxecutables/df-analyze/tree/experimental


This README covers two newer additions to **df-analyze**:

1. **Adaptive Error Rate (aER)**  
   Adaptive Error Rate (aER) provides per-sample error rate estimation for supervised machine learning models.  
   In typical laboratory validation, model performance is summarized by a single aggregate error rate over an entire dataset. While useful, such global metrics do not reflect how prediction reliability can vary substantially across individual samples.

   aER addresses this limitation by relating a model’s confidence to its expected prediction error, producing an *adaptive, sample-specific error estimate*. In a clinical setting, for example, binary cancer vs. non-cancer diagnosis, different patients may receive predictions with markedly different expected error rates. By exposing this information, aER enables clinician-users to better assess the reliability of individual predictions rather than relying solely on population-level performance statistics.

2. **Multi-target learning**  
   Multi-target learning extends `df-analyze` to support datasets with multiple target columns within a single run. This integration currently includes native multi-target support for **KNN**, **CatBoost**, and **Dummy** models.  
   Other models in the `df-analyze` framework are not yet multi-target enabled but may be extended in future releases.

## Concepts and terminology

### Adaptive Error Rate (aER)

In standard supervised classification tasks, a machine learning model typically produces two outputs for each input sample:  
(1) a predicted class label, and  
(2) an associated confidence score, usually derived from predicted class probabilities.

In conventional evaluation, model performance is summarized by a *single global error rate* (e.g., overall accuracy or misclassification rate), computed across an entire dataset. While informative at the population level, this aggregate metric obscures substantial variability in prediction reliability across individual samples.

**Adaptive Error Rate (aER)** addresses this limitation by transforming the confidence score into an estimate of the *probability that a specific prediction is incorrect*, on a per-sample basis. Formally, for a prediction made on a given sample \(x\), aER represents the expected error rate conditioned on the confidence level of that prediction:

- **aER(x)** ≈ *P(prediction is incorrect ∣ confidence level of x)*

This formulation enables error estimation at the individual-sample level rather than relying on dataset-wide averages.

Empirically, this distinction is critical in clinical and biomedical applications. Even when a dataset exhibits a low overall error rate (for example, approximately 3%), individual predictions may still carry substantially higher risk. In practice, certain cases can exhibit adaptive error rates exceeding 30%, clearly signaling increased uncertainty and reduced reliability for specific predictions. Such high-risk cases are effectively invisible when only global accuracy metrics are reported.

By exposing this sample-level variability, aER provides an explicit and interpretable estimate of prediction risk, enabling clinicians to evaluate the reliability of each diagnosis individually and supporting informed decision-making in high-stakes settings.

In contrast to standard performance metrics:

- **Overall accuracy** summarizes performance as a single dataset-level average.
- **Adaptive Error Rate (aER)** provides a sample-specific estimate of prediction risk, enabling fine-grained reliability analysis and individualized risk assessment.


### Confidence Metric

A *confidence metric* is a scalar quantity that reflects the reliability of a model’s prediction for a given sample. An effective confidence metric should vary monotonically with actual prediction accuracy.

In the context of this research, confidence metrics are primarily derived from predicted class probabilities produced by calibrated models. Within `df-analyze`, confidence metrics may be computed using:

- probability-based measures (e.g., probability margins or the difference between the highest and second-highest predicted class probabilities),
- model-specific indicators, such as vote fractions or distance-based proxies used in instance-based models like K-Nearest Neighbors.

Although confidence values are useful internally, they are often difficult to interpret directly—particularly for non-technical users—because the same numerical confidence value may correspond to very different reliability levels across models. For this reason, confidence metrics are not exposed directly as the primary reliability signal.

Instead, confidence values are systematically *mapped to expected error rates*, yielding adaptive error estimates that share a common and intuitive interpretation across models.

By default, `df-analyze` automatically selects the most effective confidence metric based on validation performance. This behavior can be overridden explicitly using the `--aer-confidence-metric` option when controlled experimentation or reproducibility requirements demand a fixed metric.

This adaptive selection process ensures that each learner employs the confidence metric best suited for accurately predicting its own error rates, thereby enhancing the interpretability and reliability of the resulting adaptive error estimates.



### Out-of-fold (OOF) predictions

To estimate aER without data leakage, `df-analyze` relies on **out-of-fold (OOF) predictions** constructed from the training split:

- the training data are divided into \(K\) folds;
- for each fold, the model is trained on \(K-1\) folds and evaluated on the held-out fold;
- predictions from all held-out folds are then concatenated.

As a result, every training sample receives a prediction from a model that **was not trained on that sample**. These OOF predictions serve as the calibration data for learning the mapping from confidence to expected error.


### Risk–coverage curve (selective prediction)

Selective prediction allows a model to **abstain** from making predictions on samples that are estimated to be unreliable.

- **Coverage** denotes the fraction of samples for which the model issues a prediction (i.e., does not abstain).
- **Risk** is the observed error rate among those accepted predictions.

By sweeping an acceptance threshold over the adaptive error rate, one obtains a **risk–coverage curve**, which explicitly characterizes the trade-off between prediction availability and reliability. As coverage decreases corresponding to rejecting higher-risk samples, the risk among accepted predictions typically decreases.

This curve enables the selection of *risk-controlled operating points*, such as enforcing policies of the form:  
“Only issue predictions when aER ≤ \(t\).”

In high-stakes applications, this provides a principled mechanism for integrating uncertainty into decision making, allowing unreliable predictions to be deferred to human review rather than being presented with misleading confidence.



### Multi-target learning (multi-output)

In this README, **multi-target learning** refers to scenarios in which:

- a dataset contains multiple target variables
- `df-analyze` trains and evaluates models for all targets within a single execution;
- target-dependent stages (e.g., feature selection, univariate analyses, and aER estimation) are performed separately for each target, then aggregated where necessary to maintain a coherent end-to-end pipeline.


## Quickstart: enabling aER and multi-target

### Enable aER (single-target classification)

```bash
python df-analyze.py   --df path/to/data.csv   --mode classify   --target outcome   --adaptive-error
```

Notes:

- aER currently runs only for **classification**.
- aER is computed **after** tuning/evaluation, using the tuned models + the train/test split that df-analyze already produced.

### 2.2 Enable multi-target learning

```bash
python df-analyze.py   --df path/to/data.csv   --mode classify   --targets outcome1,outcome2,outcome3   --classifiers knn,catboost,dummy
```

Key constraints:

- All targets must match the selected mode (all classification, or all regression).
- **Multi-target tuning is currently limited to**: **Dummy**, **KNN**, and **CatBoost** (these are the models with multi-output support in df-analyze).

### 2.3 Multi-target + aER together

aER is single-target by definition, but in multi-target runs df-analyze will run it **once per target**:

```bash
python df-analyze.py   --df path/to/data.csv   --mode classify   --targets outcome1,outcome2   --classifiers knn,catboost,dummy   --adaptive-error
```

You’ll get:

- the usual df-analyze outputs (with multi-target-aware differences), plus
- per-target aER directories:
  - `adaptive_error/outcome1/...`
  - `adaptive_error/outcome2/...`

How per-target aER is produced in multi-target runs:

- df-analyze reuses the tuned multi-target run, then **slices** predictions/probabilities per target and feeds them into the existing single-target aER engine.
- For “one estimator per target” wrappers (CatBoost/Dummy), it selects the target-specific estimator.
- For natively multi-output models (KNN), a lightweight wrapper slices the multi-output predictions down to a single target.


##  Adaptive Error Rate (aER)

### Research alignment

df-analyze’s aER implementation is meant to track the thesis "Sample-Based Error Rate Assessment Through
Predictive Confidence in Biomedical Applications" methodology closely, but it’s integrated into df-analyze’s existing pipeline (single train/test split + CV-based tuning)

The core ideas are:

1. Compute a **confidence** value (usually from probabilities).
2. Learn a mapping from confidence to expected error rate:
3. Define **Adaptive Error Rate** 
4. Estimate \(h\) non-parametrically by **binning** confidences and measuring error rates, then stabilize sparse bins via **Bayesian shrinkage** toward a global error prior.
5. Use the per-sample risk estimate for:
   - reliability checks (is risk itself calibrated?)
   - risk–coverage analysis
   - conservative, risk-controlled thresholds for abstention
   - (optionally) risk-aware ensembles
|

### How aER is integrated into the df-analyze pipeline

```
Use tuned models + train split:
  → build OOF predictions （ every row’s prediction came from a model that never saw that row in training）
  → pick calibrator + confidence metric
  → fit confidence → expected error mapping
  → score + plot on holdout test
  → write risk/coverage outputs (+ optional ensembles)
```

- aER runs only if `--adaptive-error` is set.
- aER runs only for **classification**.

aER is computed for the **top-K tuned models** (by CV score), controlled by:

- `--aer-top-k` (default: all available; set to e.g. 5 if you want to keep runtime down)

Implementation note:

- **Dummy** is skipped for aER by default. It’s a useful baseline for accuracy, but its “uncertainty” behaviour isn’t very informative for risk estimation.

Confidence metrics are only as good as the underlying probability estimates, so df-analyze can fit an **external probability calibrator** using the OOF predictions, then apply it consistently to both OOF and test computations.

Depending on what’s feasible for the model, calibrators include “none”, temperature scaling, Platt scaling, and isotonic variants. The chosen calibrator is recorded in the per-model metadata.

Unless you pass `--aer-confidence-metric`, df-analyze tries several candidate confidence metrics and picks one based on OOF behaviour:

1. compute each candidate confidence metric on OOF predictions
2. score how well it predicts correctness (a Brier-style criterion)
3. keep the metric that looks most reliable on OOF data

than for the adaptive error rate:
1. bin OOF confidence values into `--aer-bins` bins
2. compute empirical error rate per bin
3. apply Bayesian shrinkage toward the global OOF error rate (`--aer-prior-strength`)
4. optional “stability knobs”:
   - smoothing across neighbouring bins (`--aer-smooth`)
   - enforce monotonicity (`--aer-monotonic`)
   - quantile (“adaptive”) binning (`--aer-adaptive-binning`) for skewed confidence distributions

#### Evaluate the mapping on the holdout test set

1. predict labels + probabilities on test
2. apply the same calibrator chosen before 
3. compute the chosen confidence metric
4. compute per-sample **aER** 
5. write per-sample tables, binned reliability tables, and quick plots

#### Risk–coverage + a risk-controlled threshold (best model)

For the best aER-analysed model (top-ranked by df-analyze tuning), df-analyze computes:

- the **risk–coverage curve** (coverage vs accuracy under selective prediction)
- a **risk-controlled threshold** chosen conservatively using:
  - target error: `--aer-target-error`
  - significance: `--aer-alpha`
  - minimum accepted predictions: `--aer-nmin`

The threshold selection uses OOF estimates to avoid leaking information from the holdout test set.

#### Cross-model comparisons

aER also writes cross-model summary tables and comparison plots. These are handy when two models have similar accuracy/AUROC but very different reliability.

#### Ensembles

If `--aer-ensemble` is enabled, df-analyze runs risk-aware ensemble strategies. These reuse the same OOF artifacts and produce the same kinds of outputs (per-sample risk, risk–coverage curves, thresholds), under `adaptive_error/ensemble/`.


1. **Calibration via OOF predictions**  
   Instead of forcing an extra calibration split, df-analyze uses OOF predictions on the training split. This keeps the “not trained on the same sample” requirement without changing df-analyze’s basic train/test structure.

2. **Cross-fitted risk estimates when needed**  
   For risk-controlled thresholds, df-analyze can compute cross-fitted aER values on OOF data (each sample’s risk estimated by a mapping that didn’t use that sample’s fold). This is mainly about keeping things conservative.

3. **Stability on messy confidence distributions**  
   Real-world confidence distributions can be very skewed (lots of values near 1.0, sparse bins elsewhere). 
   - minimum bin counts (`--aer-min-bin-count`) and bin merging
   - Bayesian shrinkage to avoid “single sample = 0% error forever” bins
   - optional smoothing/monotonic constraints

4. **Per-target execution in multi-target mode**  
   aER stays single-target internally. Multi-target runs just slice per-target predictions and call the same engine once per target, rather than introducing an entirely new “multi-target aER” framework.

### aER configuration options

The aER stage is controlled via CLI flags. 

| Option | Meaning | Typical use |
|---|---|---|
| `--adaptive-error` | Enable aER analysis | Turn on aER outputs |
| `--aer-oof-folds K` | Folds for OOF calibration | `5` is usually fine |
| `--aer-bins B` | Number of confidence bins | `10–30` depending on data size |
| `--aer-min-bin-count N` | Minimum samples per bin | Increase on small datasets |
| `--aer-prior-strength S` | Shrinkage toward global error | Increase when bins are sparse/noisy |
| `--aer-smooth` | Smooth the expected-error curve across bins | Use if the curve looks jagged |
| `--aer-monotonic` | Enforce monotonic decrease of error with confidence | Use if noise breaks the expected shape |
| `--aer-adaptive-binning` | Use quantile bins instead of uniform bins | Helps when most confidences cluster near 1.0 |
| `--aer-confidence-metric NAME` | Fix confidence metric (skip auto-selection) | Reproducible ablations |
| `--aer-target-error ε` | Target max error for selective prediction | e.g. `0.05` (5%) |
| `--aer-alpha α` | Significance for conservative bound | e.g. `0.05` |
| `--aer-nmin N` | Minimum accepted predictions when picking a threshold | Avoid trivial thresholds |
| `--aer-top-k K` | Run aER only on top-K tuned models | Reduce runtime |
| `--aer-ensemble` | Enable ensemble strategies | Evaluate risk-aware ensembles |
| `--aer-ensemble-strategies` | Pick which ensemble strategies to run | Controlled experiments |

### aER outputs: files and figures

All aER artifacts are written under:

- **single-target:** `adaptive_error/`
- **multi-target:** `adaptive_error/<target_name>/`

---

#### Directory layout

```
adaptive_error/
  run_config.json
  plots/
    confidence_vs_expected_error_compare.png
    coverage_vs_accuracy_overlay.png              (only if --aer-ensemble)
  tables/
    models_ranked.csv
    aer_metrics_by_model.csv
    coverage_accuracy_overlay_summary.csv         (only if --aer-ensemble)
    coverage_accuracy_overlay_by_accuracy_summary.csv (only if --aer-ensemble)
  predictions/
    test_per_sample_multi_model.parquet
    test_per_sample_multi_model.csv
  models/
    <model_slug>/
      plots/
      tables/
      predictions/
      metadata/
      reports/
  ensemble/                                      (only if --aer-ensemble)
    tables/
    reports/
    <strategy_name>/
      plots/
      tables/
      predictions/
      metadata/
      reports/
```

---


##### `adaptive_error/run_config.json`

- Snapshot of the aER configuration for the run (bins, smoothing, threshold settings, etc.).
- Useful when you want to diff two runs and figure out why results moved.

##### `adaptive_error/tables/models_ranked.csv`

- “Index” of which tuned models were included in aER, with a stable `model_slug` for each.
- Ties the aER artifacts back to df-analyze’s tuning rank/metric.

You’ll typically see columns like `rank`, `model_slug`, `metric`, `cv_score`, `test_accuracy`, `n_features`, `out_dir_rel`.

##### `adaptive_error/tables/aer_metrics_by_model.csv`

- Cross-model table of reliability metrics **for the risk estimates themselves** (not just accuracy).
- Includes things like:
  - `global_error_test` (observed test error rate)
  - `brier_error_test` (MSE of predicted error probabilities)
  - `ece_error_test` (calibration error of predicted error probabilities)
- If a model’s aER stage failed, the row records a “NOT_AVAILABLE” status + reason.

##### `adaptive_error/plots/confidence_vs_expected_error_compare.png`

- Overlay of the learned confidence → expected error curves for each model.
- Quick way to spot models that are systematically over/under-confident.

##### `adaptive_error/predictions/test_per_sample_multi_model.parquet` (+ `.csv`)

- One big per-sample table for the test set, aligned across all analysed models.
- Stores, per model: predicted label, confidence, and aER.
- Handy for questions like “which samples are high-risk across *all* models?”.

##### `adaptive_error/plots/coverage_vs_accuracy_overlay.png` (only if `--aer-ensemble`)

- Overlay plot comparing the best base model’s risk–coverage curve with ensemble strategy curves.

---

#### Per-model aER : `adaptive_error/models/<model_slug>/...`

Each analysed model gets its own folder with:

- `metadata/` (JSON snapshots for reproducibility/debugging)
- `tables/` (CSV summaries)
- `plots/` (PNG diagnostics)
- `predictions/` (per-sample parquet/CSV)
- `reports/` (short Markdown summaries)

Below is what you’ll find in each subfolder.

---

### A `.../metadata/`

##### `confidence_metric_selection.json`

- Which confidence metric was chosen (or pinned), plus how alternatives compared on OOF.

##### `proba_calibrator.json`

- Which probability calibrator was used (or “none”), plus diagnostics from the OOF selection step.

##### `confidence_to_expected_error_lookup.json`

- The fitted lookup defining \(h(c)\): confidence → expected error.
- This is the core thing you’d reuse if you wanted to apply aER on new data without refitting.

##### `adaptive_error_metrics.json`

- Test-set summary metrics evaluating the quality/calibration of the aER estimates.

##### `risk_control_threshold.json` (best model only)

- The chosen threshold \(t^*\) and coverage/guarantee details for risk-controlled selective prediction.

##### `sanity_checks.json` (only when issues detected)

- A grab bag of automatically detected warnings (confidence weirdness, calibration drift, etc.) that suggest you should treat the result cautiously.

##### `error_traceback.txt` (only if a stage fails)

- Captured exception tracebacks, so failures are inspectable without killing the full df-analyze run.

---

### B `.../tables/`

##### `confidence_to_expected_error_lookup.csv`

- CSV version of the lookup table for quick inspection/plotting.

##### `oof_confidence_error_bins.csv`

- The binned OOF stats used to fit \(h\). Typically includes bin edges, counts, empirical error, shrunken/smoothed expected error, and uncertainty bounds (e.g. Wilson intervals).

##### `test_confidence_error_bins.csv`

- Same idea as above, but computed on the **test set** using the learned mapping.

##### `test_error_reliability_bins.csv`

- Reliability of **aER itself as a probability forecast of error**:
  - bins are by predicted error probability
  - compare mean predicted error vs empirical error in each bin

##### `coverage_accuracy_curve.csv` (best model only)

- Risk–coverage curve data (coverage, accuracy, thresholds, etc.). Generated by sorting samples by aER (low risk first) and sweeping acceptance.

##### `coverage_summary.csv` (best model only)

- A condensed set of operating points (selected coverages + their accuracy/risk).

##### `clinician_view.csv` (best model only)

A clinician-facing per-sample table for manual audit. Usually includes:

- `row_id`
- true label (if available)
- predicted label + decoded label (`y_pred_label`)
- `aer` / `aer_pct`
- a flag like `flag_gt_target_error`

##### `top20_highest_adaptive_error.csv` (best model only)

- The 20 highest-risk test predictions (sorted by aER descending). Good for “worst-case” audits.

---

### C `.../plots/`

##### `confidence_vs_expected_error.png`

- Single-model plot of confidence vs expected error.
- Sanity checks you usually want:
  - generally decreasing shape
  - not wildly jagged (unless data is tiny)

##### `coverage_vs_accuracy.png` (best model only)

- Plot version of the risk–coverage curve.

---

### D `.../predictions/`

##### `oof_per_sample.parquet` (+ `.csv`)

- The raw per-sample OOF table used for calibration. Typical columns include:
  - `row_id`, `fold`
  - `y_true`, `y_pred_oof`
  - candidate confidence metrics
  - calibrated probabilities (parquet)
  - `aer` and `aer_cv` (when available)

##### `test_per_sample.parquet` (+ `.csv`)

- The main per-sample artifact for the test set: predictions + confidence diagnostics + aER. Usually includes:
  - `row_id`
  - `y_true`, `y_pred`, `correct`
  - `confidence`
  - `aer`, `aer_pct`
  - `flag_gt_target_error`
  - decoded labels (`y_true_label`, `y_pred_label`)
  - `y_pred_from_cal_proba` (argmax from calibrated probs; mostly a sanity check)
  - probability diagnostics from calibrated probs:
    - `p_max`, `p_2nd`, `p_margin`
    - `p_pred`, `p_pred_margin`

---

### E `.../reports/`

Markdown summaries for humans (mirrors the key CSV tables):

- `clinician_view.md` (best model only)
- `coverage_summary.md` (best model only)
- `risk_control_threshold.md` (best model only)

---

#### Ensemble outputs (optional): `adaptive_error/ensemble/`

Ensemble analysis is enabled with `--aer-ensemble`.

##### `adaptive_error/ensemble/tables/ensemble_summary.csv`

- One table comparing ensemble strategies on accuracy/risk/coverage + calibration metrics.

##### `adaptive_error/ensemble/reports/ensemble_summary.md`

- Markdown version of the same summary.

##### `adaptive_error/ensemble/<strategy_name>/...`

A per-strategy directory that mirrors the per-model structure:

- `metadata/strategy.json`: strategy definition + parameters
- `metadata/hens_calibrator.json`: calibrator metadata where applicable
- `predictions/cv_ensemble.parquet`: CV ensemble outputs used for risk calibration
- `predictions/test_per_sample.*`: per-sample test predictions and aER for the ensemble
- the same reliability tables and plots as base models

---

## Multi-target learning

### How multi-target is integrated into df-analyze

Multi-target support was added by extending df-analyze’s existing abstractions

1. CLI accepts multiple targets (`--targets`)
2. Prepared data stores `y` as a DataFrame when multiple targets are provided
3. Target-dependent stages are reused by slicing (`PreparedData.for_target(...)`)
4. Feature selection runs per target, then is aggregated into a shared feature set
5. Models can train/evaluate with multi-output `y`, and reporting stays per-target

##### CLI: `--targets`

- `--target` still exists (single-target, backwards compatible).
- `--targets y1,y2,...` activates multi-target mode 


### Multi-target feature selection aggregation

Multi-target selection is basically two steps:

1. Run selection per target 
2. Combine those per-target outputs into a shared feature list.

Two aggregation strategies are supported:

#### A `--mt-agg-strategy borda` (default)

A rank aggregation approach:

- each target yields a ranked feature list
- ranks are combined across targets
- features that show up in more targets get favoured via `--mt-agg-alpha` (support weighting exponent)

This is a decent default when targets are noisy but you want one stable shared subset.

#### B `--mt-agg-strategy freq`

A simpler frequency-based approach:

- count how often each feature was selected across targets
- keep features that show up often

This ignores per-target ranking scores, but it’s easy to reason about.

#### Support and size controls

- `--mt-min-support`: minimum fraction of targets that must include a feature
- `--mt-top-k`: maximum number of aggregated features to keep

These help keep models tractable and results interpretable.

---

### Multi-target models

Multi-target training is currently supported for:

- **KNN** (native multi-output support in scikit-learn)
- **Dummy** (implemented as one estimator per target)
- **CatBoost** (implemented as one estimator per target)



## Output changes in multi-target mode

Multi-target keeps df-analyze’s overall directory layout, but two things change consistently:

1. **Target-dependent stages write to per-target subdirectories.**
2. **Final evaluation tables become target-aware.**

### Per-target subdirectories

With `--targets`, target-dependent artifacts are written under paths like:

- `features/associations/<target>/...`
- `features/predictions/<target>/...`
- `selection/<target>/...`

These contain the same kinds of files as the single-target pipeline, just repeated once per target.

### Aggregated selection 

After aggregating per-target selections, df-analyze writes the aggregated results in the usual places:

- `selection/filter/...` (aggregated filter results)
- `selection/embed/...` and `selection/wrapper/...` (aggregated embedded/wrapper results)

Aggregated JSON payloads include `target_names` metadata so you can see which targets were involved.

### Prepared data 

In `prepared/` (and in train/test prepared folders):

- `y.parquet`, `y_train.parquet`, `y_test.parquet` contain **multiple columns** (one per target).
- for classification, `labels.json` becomes nested (per target).

Everything else is structurally the same.

### Results tables

In `results/`:

- `performance_long_table.csv` gains a `target` column.
- Markdown summaries expand accordingly (each model/selection pair appears once per target).





