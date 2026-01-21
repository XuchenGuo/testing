# df-analyze — Adaptive Error Rate (aER) + Multi‑Target Learning Integration

This README covers two newer additions to **df-analyze**:

1. **Adaptive Error Rate (aER)**: a calibration/reliability layer that gives you *per-sample expected error estimates*, *risk–coverage curves*, and *risk-controlled thresholds* for selective prediction.
2. **Multi-target learning**: run df-analyze on **multiple target columns** in one go, including multi-target support for **KNN**, **CatBoost**, and **Dummy**.

This file is not meant to replace your “main” df-analyze README. It’s just the delta: what changed, how it plugs into the existing pipeline, and what new output artifacts to expect.

---

<!-- omit from toc -->
## Contents

- [1. Concepts and terminology](#1-concepts-and-terminology)
- [2. Quickstart: enabling aER and multi-target](#2-quickstart-enabling-aer-and-multi-target)
- [3. Adaptive Error Rate (aER)](#3-adaptive-error-rate-aer)
  - [3.1 Research alignment](#31-research-alignment)
  - [3.2 How aER is integrated into the df-analyze pipeline](#32-how-aer-is-integrated-into-the-df-analyze-pipeline)
  - [3.3 Practical adaptations made for df-analyze](#33-practical-adaptations-made-for-df-analyze)
  - [3.4 aER configuration options](#34-aer-configuration-options)
  - [3.5 aER outputs: files and figures](#35-aer-outputs-files-and-figures)
- [4. Multi-target learning](#4-multi-target-learning)
  - [4.1 How multi-target is integrated into df-analyze](#41-how-multi-target-is-integrated-into-df-analyze)
  - [4.2 Multi-target feature selection aggregation](#42-multi-target-feature-selection-aggregation)
  - [4.3 Multi-target models](#43-multi-target-models)
- [5. Output changes in multi-target mode](#5-output-changes-in-multi-target-mode)
- [6. Developer map: where the integrations live](#6-developer-map-where-the-integrations-live)

---

## 1. Concepts and terminology

### Adaptive Error Rate (aER)

In a typical classification run, a model gives you:

- a predicted label \(\hat{y}(x)\), and
- some “confidence-ish” signal (usually derived from probabilities).

**Adaptive Error Rate (aER)** takes that confidence signal and turns it into an *estimated probability of being wrong*, per sample:

- **aER(x)** ≈ “For predictions that look like this, how often is the model incorrect?”

This is different from accuracy:

- **accuracy** is an average over a dataset
- **aER** is a per-sample risk estimate, mainly for *reliability diagnostics* and *selective prediction* (abstaining when the risk looks too high)

### Confidence metric

A “confidence metric” here is just a scalar that moves in the right direction as predictions get more certain.

In the thesis framing, confidence usually comes from predicted probabilities (binary or multiclass). In df-analyze, confidence can be:

- probability-derived (e.g. probability margin),
- KNN-flavoured (e.g. vote fraction, neighbour-distance proxy),
- and model-specific signals where it makes sense (kept mainly so the interface stays consistent even if you swap models).

By default df-analyze picks a confidence metric for you (see §3.2 Step 4), unless you pin it with `--aer-confidence-metric`.

### Out-of-fold (OOF) predictions

To estimate aER without leakage, df-analyze builds **out-of-fold predictions** on the training split:

- split training data into K folds
- for each fold: train on K−1 folds, predict on the held-out fold
- concatenate those held-out predictions

That way, every training sample gets a prediction from a model that **did not train on that sample**. Those OOF predictions act as the calibration data for learning the mapping from confidence → expected error.

### Risk–coverage curve (selective prediction)

Selective prediction means the model is allowed to **abstain** on uncertain cases.

- **Coverage**: fraction of samples you *do* predict on (don’t abstain)
- **Risk**: error rate among the samples you accepted

Sweeping an acceptance threshold gives a **risk–coverage** curve, and you can pick risk-controlled operating points (“predict only when aER ≤ t”).

### Multi-target learning (multi-output)

In this README, **multi-target** means:

- you have multiple targets \(y_1, y_2, \dots, y_T\)
- df-analyze trains and evaluates in one run, reporting per-target results
- target-dependent stages (selection, univariate reports, aER) are run per target, and then combined where needed so the pipeline stays coherent

This is multi-output learning. It doesn’t assume your targets are dependent; for some models (CatBoost/Dummy) df-analyze simply trains one model per target under a single wrapper.

---

## 2. Quickstart: enabling aER and multi-target

### 2.1 Enable aER (single-target classification)

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

The goal is to keep the aER implementation itself simple/single-target while still supporting multi-target datasets.

---

## 3. Adaptive Error Rate (aER)

### 3.1 Research alignment

df-analyze’s aER implementation is meant to track the thesis methodology closely, but it’s integrated into df-analyze’s existing pipeline (single train/test split + CV-based tuning), so a few practical compromises show up later in §3.3.

It aligns with (naming as in the project):

- **Xuchen_Guo_MSc_Thesis** (definitions/assumptions/method)
- **my_thesis_topic_oooo** (motivation + “clinician-facing” output style)

At a high level, the structure mirrors:

- **Thesis §2.1.2**: probability-derived confidence metrics (binary + multiclass)
- **Thesis §2.1.3**: aER definition \(\mathrm{aER}(x)\) and the confidence→error mapping \(h(c)\), estimated via binning + a global-error prior
- **Thesis §2.1.4–2.1.5**: selective prediction + risk-controlled thresholds using conservative bounds
- **Thesis §2.6**: ensemble extensions driven by risk estimates

The `clinician_view.*` and “high-risk case” outputs are included specifically to support the intended “flag risky predictions for review” workflow described in the thesis topic document.

#### 3.1.1 What the thesis defines (conceptually)

The core ideas are:

1. Compute a **confidence** value (usually from probabilities).
2. Learn a mapping \(h(c)\) from confidence \(c\) to empirical error probability:
   - \(h(c) \approx \Pr(\hat{y}(x) \neq y(x) \mid c(x) = c)\)
3. Define **Adaptive Error Rate** as:
   - \(\text{aER}(x) = h(c(x))\)
4. Estimate \(h\) non-parametrically by **binning** confidences and measuring error rates, then stabilize sparse bins via **Bayesian shrinkage** toward a global error prior.
5. Use the per-sample risk estimate for:
   - reliability checks (is risk itself calibrated?)
   - risk–coverage analysis
   - conservative, risk-controlled thresholds for abstention
   - (optionally) risk-aware ensembles

#### 3.1.2 How df-analyze implements those same ideas

Here are the rough 1:1 correspondences in the codebase:

| Thesis concept | df-analyze implementation |
|---|---|
| Confidence from probabilities | `analysis/adaptive_error/confidence_metrics.py` implements probability-derived confidence metrics (plus some model-specific options). |
| Mapping \(h(c)\) via binned error rates | `analysis/adaptive_error/aer.py` (`AdaptiveErrorCalculator`) builds the confidence→expected-error lookup via bins. |
| Bayesian shrinkage with global error prior | `AdaptiveErrorCalculator` supports a **prior strength** that pulls sparse bins toward the global error rate. |
| Calibration separated from training | df-analyze uses **OOF predictions** on the training split as a leakage-safe calibration proxy. |
| Risk–coverage / selective prediction | `analysis/adaptive_error/risk_control.py` plus plotting in `analysis/adaptive_error/plots.py`. |
| Risk-controlled thresholds | `risk_control.find_threshold(...)` picks a threshold using an upper confidence bound criterion. |
| Ensemble extensions | `analysis/adaptive_error/ensemble_*` implements thesis-aligned strategy families. |

### 3.2 How aER is integrated into the df-analyze pipeline

aER is implemented as a **post-evaluation stage**. It doesn’t fork df-analyze into a separate training pipeline; it consumes the stuff df-analyze already computed.

Roughly:

```
(df-analyze core)
Data → Prepare → (feature selection) → Tune → Evaluate on holdout

(aER extension)
Use tuned models + train split:
  → build OOF predictions
  → pick calibrator + confidence metric
  → fit confidence → expected error mapping
  → score + plot on holdout test
  → write risk/coverage outputs (+ optional ensembles)
```

Below is the same idea broken down into the steps you’ll see in the implementation.

#### Step 0 — Entry point and gating

- aER runs only if `--adaptive-error` is set.
- aER runs only for **classification**.

Hook point: after `evaluate_tuned(...)` finishes, df-analyze calls the aER runner (see `df_analyze/_main.py`).

#### Step 1 — Choose which tuned models to analyze

aER is computed for the **top-K tuned models** (by CV score), controlled by:

- `--aer-top-k` (default: all available; set to e.g. 5 if you want to keep runtime down)

Implementation note:

- **Dummy** is skipped for aER by default. It’s a useful baseline for accuracy, but its “uncertainty” behaviour isn’t very informative for risk estimation.

#### Step 2 — Build OOF prediction tables (calibration proxy)

For each selected tuned model:

1. reuse the training split \((X_{train}, y_{train})\)
2. run a K-fold OOF loop (`--aer-oof-folds`) using df-analyze’s splitter logic (same grouping constraints, etc.)
3. write an **OOF per-sample table** containing:
   - true label
   - out-of-fold predicted label
   - raw probabilities/scores (when available)
   - candidate confidence metrics derived from those probabilities/scores

This is the “no cheating” calibration set: every row’s prediction came from a model that never saw that row in training.

#### Step 3 — Probability calibration (optional, but usually a good idea)

Confidence metrics are only as good as the underlying probability estimates, so df-analyze can fit an **external probability calibrator** using the OOF predictions, then apply it consistently to both OOF and test computations.

Depending on what’s feasible for the model/setting, calibrators include “none”, temperature scaling, Platt scaling, and isotonic variants. The chosen calibrator is recorded in the per-model metadata.

#### Step 4 — Pick a confidence metric (automatic unless you pin it)

Unless you pass `--aer-confidence-metric`, df-analyze tries several candidate confidence metrics and picks one based on OOF behaviour:

1. compute each candidate confidence metric on OOF predictions
2. score how well it predicts correctness (a Brier-style criterion)
3. keep the metric that looks most reliable on OOF data

The selection is written to disk so you can review what happened.

#### Step 5 — Fit the confidence → expected error mapping

This is where \(h(c)\) is fit, using `AdaptiveErrorCalculator`:

1. bin OOF confidence values into `--aer-bins` bins
2. compute empirical error rate per bin
3. apply Bayesian shrinkage toward the global OOF error rate (`--aer-prior-strength`)
4. optional “stability knobs”:
   - smoothing across neighbouring bins (`--aer-smooth`)
   - enforce monotonicity (`--aer-monotonic`)
   - quantile (“adaptive”) binning (`--aer-adaptive-binning`) for skewed confidence distributions

The output is effectively a lookup function: confidence in → expected error out.

#### Step 6 — Evaluate the mapping on the holdout test set

On \((X_{test}, y_{test})\) using the tuned model trained on the full train split:

1. predict labels + probabilities on test
2. apply the same calibrator chosen in Step 3
3. compute the chosen confidence metric
4. compute per-sample **aER** via the fitted lookup \(h\)
5. write per-sample tables, binned reliability tables, and quick plots

#### Step 7 — Risk–coverage + a risk-controlled threshold (best model)

For the best aER-analysed model (top-ranked by df-analyze tuning), df-analyze computes:

- the **risk–coverage curve** (coverage vs accuracy under selective prediction)
- a **risk-controlled threshold** \(t^*\) (“accept only when aER ≤ \(t^*\)”), chosen conservatively using:
  - target error: `--aer-target-error`
  - significance: `--aer-alpha`
  - minimum accepted predictions: `--aer-nmin`

The threshold selection uses OOF estimates to avoid leaking information from the holdout test set.

#### Step 8 — Cross-model comparisons

aER also writes cross-model summary tables and comparison plots. These are handy when two models have similar accuracy/AUROC but very different reliability.

#### Optional Step 9 — Ensembles

If `--aer-ensemble` is enabled, df-analyze runs risk-aware ensemble strategies. These reuse the same OOF artifacts and produce the same kinds of outputs (per-sample risk, risk–coverage curves, thresholds), under `adaptive_error/ensemble/`.

### 3.3 Practical adaptations made for df-analyze

The thesis can assume a clean train/calibration/test split. df-analyze can’t always (it’s an AutoML pipeline that starts from “here’s one dataset”), so the integration makes a few pragmatic choices:

1. **Calibration via OOF predictions**  
   Instead of forcing an extra calibration split, df-analyze uses OOF predictions on the training split. This keeps the “not trained on the same sample” requirement without changing df-analyze’s basic train/test structure.

2. **Cross-fitted risk estimates when needed**  
   For risk-controlled thresholds, df-analyze can compute cross-fitted aER values on OOF data (each sample’s risk estimated by a mapping that didn’t use that sample’s fold). This is mainly about keeping things conservative.

3. **Stability on messy confidence distributions**  
   Real-world confidence distributions can be very skewed (lots of values near 1.0, sparse bins elsewhere). To keep \(h\) from doing something silly, df-analyze supports:
   - minimum bin counts (`--aer-min-bin-count`) and bin merging
   - Bayesian shrinkage to avoid “single sample = 0% error forever” bins
   - optional smoothing/monotonic constraints

4. **Per-target execution in multi-target mode**  
   aER stays single-target internally. Multi-target runs just slice per-target predictions and call the same engine once per target, rather than introducing an entirely new “multi-target aER” framework.

### 3.4 aER configuration options

The aER stage is controlled via CLI flags. The ones you’ll typically care about:

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
| `--aer-ensemble-strategies ...` | Pick which ensemble strategies to run | Controlled experiments |

### 3.5 aER outputs: files and figures

All aER artifacts are written under:

- **single-target:** `adaptive_error/`
- **multi-target:** `adaptive_error/<target_name>/`

The layout follows the same pattern as df-analyze’s other pipeline stages: predictable folders, machine-readable tables, and a few Markdown summaries.

A design detail worth calling out: when a stage fails for a specific model, df-analyze writes placeholder “NOT_AVAILABLE” artifacts rather than silently skipping outputs. That way downstream tooling doesn’t break mysteriously.

---

#### 3.5.1 Directory layout

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

#### 3.5.2 Root-level aER artifacts

These summarize *all* analysed models for one target.

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

#### 3.5.3 Per-model aER artifacts: `adaptive_error/models/<model_slug>/...`

Each analysed model gets its own folder with:

- `metadata/` (JSON snapshots for reproducibility/debugging)
- `tables/` (CSV summaries)
- `plots/` (PNG diagnostics)
- `predictions/` (per-sample parquet/CSV)
- `reports/` (short Markdown summaries)

Below is what you’ll find in each subfolder.

---

### A) `.../metadata/`

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

### B) `.../tables/`

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

### C) `.../plots/`

##### `confidence_vs_expected_error.png`

- Single-model plot of confidence vs expected error.
- Sanity checks you usually want:
  - generally decreasing shape
  - not wildly jagged (unless data is tiny)

##### `coverage_vs_accuracy.png` (best model only)

- Plot version of the risk–coverage curve.

---

### D) `.../predictions/`

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

### E) `.../reports/`

Markdown summaries for humans (mirrors the key CSV tables):

- `clinician_view.md` (best model only)
- `coverage_summary.md` (best model only)
- `risk_control_threshold.md` (best model only)

---

#### 3.5.4 Ensemble outputs (optional): `adaptive_error/ensemble/`

Ensemble analysis is enabled with `--aer-ensemble`.

At a high level you’ll see:

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

The point is to let you evaluate thesis-motivated ideas directly: do risk-aware ensembles improve selective prediction, and do their risk estimates stay calibrated?

---

## 4. Multi-target learning

### 4.1 How multi-target is integrated into df-analyze

Multi-target support was added by extending df-analyze’s existing abstractions, not by bolting on a second pipeline.

At a high level:

1. CLI accepts multiple targets (`--targets`)
2. Prepared data stores `y` as a DataFrame when multiple targets are provided
3. Target-dependent stages are reused by slicing (`PreparedData.for_target(...)`)
4. Feature selection runs per target, then is aggregated into a shared feature set
5. Models can train/evaluate with multi-output `y`, and reporting stays per-target

Design principle:

> Multi-target should feel like “the same pipeline, repeated where the pipeline is inherently target-dependent”, not a separate mode with different code paths everywhere.

#### Step-by-step integration

##### Step 0 — CLI: `--targets` extends `--target`

- `--target` still exists (single-target, backwards compatible).
- `--targets y1,y2,...` activates multi-target mode and overrides `--target`.

There isn’t a separate entry point; it’s still the same `main()` flow with a target list.

---

##### Step 1 — Preparation stage: multi-target `y`

In multi-target mode:

- `PreparedData.y` is a **DataFrame** with one column per target.
- For classification, `labels.json` becomes nested:
  - `{target_name: {encoded_int: original_label_string}}`

This keeps the rest of the pipeline sane: label decoding remains correct per target, and you can still slice to a single-target view when needed.

---

##### Step 2 — Train/test splitting: stratification with multiple targets

Classification wants stratified splits, but multi-target introduces a nasty edge case: the joint combinations of target levels can explode, producing lots of rare combinations.

df-analyze handles this by building a **combined stratification label** from multiple targets, and (if needed) dropping the most unique target(s) until the split is feasible.

It’s not perfect, but it preserves the intent of stratification without inventing a whole new sampling system.

---

##### Step 3 — Univariate analysis and feature selection: reuse by slicing

Most univariate analyses and selection methods are inherently single-target. Rather than rewrite them, df-analyze does:

1. iterate over targets
2. call `PreparedData.for_target(target)` to get a single-target view
3. run the existing routines unchanged
4. write outputs under per-target subdirectories

This is the main “reuse instead of rewrite” trick.

---

##### Step 4 — Aggregate selected features across targets

After per-target selection, df-analyze aggregates features into one shared feature set (see §4.2). That shared set is what gets passed into tuning/evaluation so the data flow stays consistent.

---

##### Step 5 — Tuning and evaluation with multi-output `y`

The model interface is extended so models can accept:

- `y_train` as a Series (single-target) **or**
- `y_train` as a DataFrame (multi-target)

For tuning:

- a single Optuna trial proposes one parameter set
- the objective is the **mean score across targets**

Runtime control (important in multi-target runs):

- df-analyze reuses your global `--htune-trials` budget but scales it down so multi-target runs don’t blow up:
  - effective trials ≈ `max(15, htune_trials // n_targets)`  
  - CatBoost gets an extra cap for stability

For reporting:

- the long performance table gains a `target` column
- Markdown summaries expand naturally (you’ll see one row per target)

No “parallel tuner” is introduced; it’s the same tuning stage.

---

### 4.2 Multi-target feature selection aggregation

Multi-target selection is basically two steps:

1. Run selection per target (unchanged methods).
2. Combine (“aggregate”) those per-target outputs into a shared feature list.

Two aggregation strategies are supported:

#### A) `--mt-agg-strategy borda` (default)

A rank aggregation approach:

- each target yields a ranked feature list
- ranks are combined across targets
- features that show up in more targets get favoured via `--mt-agg-alpha` (support weighting exponent)

This is a decent default when targets are noisy but you want one stable shared subset.

#### B) `--mt-agg-strategy freq`

A simpler frequency-based approach:

- count how often each feature was selected across targets
- keep features that show up often

This ignores per-target ranking scores, but it’s easy to reason about.

#### Support and size controls

- `--mt-min-support`: minimum fraction of targets that must include a feature
- `--mt-top-k`: maximum number of aggregated features to keep

These help keep models tractable and results interpretable.

---

### 4.3 Multi-target models

Multi-target training is currently supported for:

- **KNN** (native multi-output support in scikit-learn)
- **Dummy** (implemented as one estimator per target)
- **CatBoost** (implemented as one estimator per target)

#### 4.3.1 Shared design: extend the model interface, not the pipeline

The key change is in `df_analyze/models/base.py`:

- `DfAnalyzeModel.fit(...)` accepts `y` as a Series **or** DataFrame.
- `predict(...)` returns:
  - a Series (single-target), or
  - a DataFrame (multi-target, one column per target).
- `predict_proba(...)` returns:
  - a single probability array (single-target), or
  - a dict `{target_name: proba_array}` (multi-target) where appropriate.

So the tuning/eval pipeline can call the same methods; the model wrappers normalize I/O shapes.

---

#### 4.3.2 KNN (multi-target)

**Where:** `df_analyze/models/knn.py`

- scikit-learn KNN estimators accept multi-output `y` directly.
- df-analyze passes the multi-target DataFrame into `.fit(...)`.
- `predict(...)` returns an \(n \times T\) array, which df-analyze wraps into a DataFrame.
- For multi-output classification, scikit-learn returns a **list of probability arrays** (one per target) from `predict_proba(...)`; df-analyze turns that into a dict keyed by target name.

---

#### 4.3.3 Dummy (multi-target)

**Where:** `df_analyze/models/dummy.py`

Dummy is a baseline, and multi-output support across dummy variants isn’t consistent enough to rely on. df-analyze therefore implements Dummy as “one model per target”:

- `self.models[target] = DummyClassifier(...)` / `DummyRegressor(...)`
- `.fit(...)` loops over targets and fits each model
- `.predict(...)` loops and concatenates into a DataFrame
- `.predict_proba(...)` returns `{target: proba}`

Treating targets independently here keeps the baseline meaning clean (and avoids accidental coupling).

---

#### 4.3.4 CatBoost (multi-target)

**Where:** `df_analyze/models/catboost.py`

CatBoost doesn’t expose a single “multi-output classifier” interface that fits neatly into df-analyze’s tuning/evaluation flow, so it follows the same pattern as Dummy:

- train one CatBoost model per target:
  - `self.models[target] = CatBoostClassifier(...)` / `CatBoostRegressor(...)`

Categorical features:

- df-analyze uses CatBoost’s native categorical handling
- categorical indices come from the prepared data’s `X_cat` / `X` representation

Compute notes:

- CatBoost will use GPU if available; df-analyze falls back to CPU safely if GPU isn’t available or fails.
- Multi-target tuning can get expensive quickly, so the multi-target model set is kept small and trials are scaled down (see §4.1 Step 5).

---

## 5. Output changes in multi-target mode

Multi-target keeps df-analyze’s overall directory layout, but two things change consistently:

1. **Target-dependent stages write to per-target subdirectories.**
2. **Final evaluation tables become target-aware.**

### 5.1 Per-target subdirectories

With `--targets`, target-dependent artifacts are written under paths like:

- `features/associations/<target>/...`
- `features/predictions/<target>/...`
- `selection/<target>/...`

These contain the same kinds of files as the single-target pipeline, just repeated once per target.

### 5.2 Aggregated (shared) selection artifacts

After aggregating per-target selections, df-analyze writes the aggregated results in the usual places:

- `selection/filter/...` (aggregated filter results)
- `selection/embed/...` and `selection/wrapper/...` (aggregated embedded/wrapper results)

Aggregated JSON payloads include `target_names` metadata so you can see which targets were involved.

### 5.3 Prepared data artifacts that change meaning

In `prepared/` (and in train/test prepared folders):

- `y.parquet`, `y_train.parquet`, `y_test.parquet` contain **multiple columns** (one per target).
- for classification, `labels.json` becomes nested (per target).

Everything else is structurally the same.

### 5.4 Results tables become target-aware

In `results/`:

- `performance_long_table.csv` gains a `target` column.
- Markdown summaries expand accordingly (each model/selection pair appears once per target).

This matters for review: tuning optimizes an averaged objective, but reporting stays transparent per target.

---

## 6. Developer map: where the integrations live

If you’re maintaining/extending these pieces, the key modules are:

### Adaptive Error Rate (aER)

- `df_analyze/analysis/adaptive_error/runner.py`  
  Orchestrates aER execution from prepared data and tuned results.
- `df_analyze/analysis/adaptive_error/oof.py` + `oof_stage.py`  
  Builds OOF predictions, selects probability calibrator and confidence metric.
- `df_analyze/analysis/adaptive_error/aer.py`  
  `AdaptiveErrorCalculator`: fits the confidence→expected error mapping.
- `df_analyze/analysis/adaptive_error/test_stage.py`  
  Applies mapping on test data, writes per-sample outputs and diagnostics.
- `df_analyze/analysis/adaptive_error/risk_control.py` + `risk_control_writer.py`  
  Risk-controlled threshold selection and reporting.
- `df_analyze/analysis/adaptive_error/ensemble_*`  
  Optional thesis-aligned ensemble extensions.

### Multi-target learning

- `df_analyze/cli/cli.py`  
  Adds `--targets` and multi-target aggregation options; maps them into `ProgramOptions`.
- `df_analyze/preprocessing/prepare.py`  
  Stores multi-target `y` as a DataFrame; adds `PreparedData.for_target(...)`.
- `df_analyze/splitting.py`  
  Implements multi-target-aware stratification labels (`y_split_label`).
- `df_analyze/selection/multitarget.py`  
  Aggregates per-target feature selection results.
- `df_analyze/models/base.py`  
  Extends the model interface to accept/return multi-target structures.
- `df_analyze/models/knn.py`, `models/dummy.py`, `models/catboost.py`  
  Multi-target model implementations.

### Integration wiring

- `df_analyze/_main.py`  
  Runs per-target univariate/selection via `PreparedData.for_target(...)`, aggregates selections, runs multi-target tuning, and (if enabled) runs aER per target by slicing evaluation results.

---

## Appendix: How to read the aER outputs scientifically

If you only have time for a quick pass:

1. Start with `adaptive_error/tables/aer_metrics_by_model.csv` and pick a couple of promising models (good accuracy *and* decent aER calibration).
2. For the best model:
   - open `models/<slug>/plots/confidence_vs_expected_error.png` (basic sanity)
   - check `models/<slug>/tables/test_error_reliability_bins.csv` (is aER itself calibrated?)
   - look at `models/<slug>/plots/coverage_vs_accuracy.png` (selective prediction behaviour)
   - read `models/<slug>/reports/risk_control_threshold.md` (what threshold was chosen and what it implies)
3. For case-level audits, `clinician_view.csv` and `top20_highest_adaptive_error.csv` are the fastest “show me the risky stuff” entry points.

That workflow matches the intent of the thesis docs: turn raw model confidence into a reviewable, per-sample risk signal you can actually act on.
