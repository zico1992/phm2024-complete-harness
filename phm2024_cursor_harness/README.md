# Modelling reference

New to this repo? Read the [project README](../README.md) first, then [START_HERE.md](START_HERE.md). This file is the detailed modelling reference: features, splits, candidates, and run outputs.

The consolidated workflow is in [START_HERE.md](START_HERE.md): EDA through runtime-checked
frozen predictions, delayed labels and prioritized reviews. Preserve existing completed runs.

# Current phase: frozen-baseline review pilot

The prediction interface, reliability checks, local review dashboard and labelled operational
shadow workflow are implemented. Start with [PILOT_GUIDE.md](PILOT_GUIDE.md) and the updated
[CURSOR_TASK.md](CURSOR_TASK.md). [OPERATIONAL_DATA_REQUEST.md](OPERATIONAL_DATA_REQUEST.md)
lists the missing operational inputs. The included demo uses the frozen SAMPLE model;
your full-run selected_model.joblib must be registered locally. No model was retrained.

# PHM 2024 Cursor modelling harness

An executable project for EDA, physics-informed/statistical features, feature selection,
joint engine-health classification and probabilistic torque-margin regression, calibration,
internal validation, operating-regime stress tests and challenge submission export.

Open this folder in Cursor and use CURSOR_TASK.md. AGENTS.md and .cursor/rules/phm.mdc
contain the modelling contracts. This is ordinary Python; no paid service or LLM is required.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

On macOS/Linux, use python3 -m venv .venv and source .venv/bin/activate.
If the archive includes data/, the four supplied CSVs are already in the correct place.
Otherwise copy X_train.csv, y_train.csv, X_test.csv, X_validation.csv into data/.
Commands below run from this project folder, not from inside phm/.

## Run

```powershell
# Complete descriptive EDA on every supplied row; no models.
python run.py --stage eda --data-dir data --output-dir runs/eda

# Quick, reproducible experiment on 60,000 training rows.
python run.py --config configs/default.yaml --data-dir data --output-dir runs/quick

# Final experiment on ALL 742,625 training rows; higher CPU/RAM cost.
python run.py --config configs/full.yaml --data-dir data --output-dir runs/full

# Verify split integrity, saved-model reproducibility and submission schema/coverage.
python verify_run.py --run-dir runs/full --data-dir data

# Labelled physics hypothesis experiment: PA in metres, IAS in knots.
# Do not use this as confirmed physics without metadata verification.
python run.py --config configs/altitude_m_hypothesis.yaml --data-dir data --output-dir runs/altitude_m

# Reproduce predictions from the exact saved/evaluated model.
python run.py --stage predict --model runs/full/selected_model.joblib --data-dir data --output-dir runs/predict_full
```

Use --max-rows 20000 for a smaller smoke experiment, or --max-rows 0 for all rows.
--skip-stress suppresses the additional regime-exclusion refits; record that limitation.
Every run directory must be new/empty. The log and incremental candidate metrics aid debugging.
Only load joblib artifacts from trusted sources.

## Data-specific observations

The attached files have 742,625 labelled training rows, 21,436 test rows and 21,436
validation rows. All seven predictors are numeric, with no missing values. There are
59,600 repeated training feature rows. Negative margin is not a deterministic fault
definition: healthy and faulty labels both occur on either side of zero.

The original source and published papers disagree on PA semantics. The conservative
default keeps PA opaque and labels empirical ratios as proxies. Altitude derivations
are opt-in; temperature assumptions are explicit. See FEATURE_CATALOG.md.

## Validation design

Approximate row proportions (actual counts vary because duplicates stay together):

| Partition | Share | Purpose |
|---|---:|---|
| Train | 60% | Fit features and candidate models; group-aware OOF margin stacking |
| Selection | 15% | Compare candidates and selected-feature ablation |
| Calibration | 10% | Sigmoid probability calibration, PDF-scale correction, F2 threshold |
| Final holdout | 15% | Evaluate the frozen model once |

Final fitting combines train+selection (about 75%), fits a fresh OOF stack, then calibrates
on calibration only. Final holdout never enters the saved model. Keeping the same evaluated
artifact for submissions makes reported results auditable; it intentionally does not refit
on the holdout. Subsampling is target-independent and occurs before split creation.
All learned preprocessing is fit on the current fitting partition or auxiliary OOF fold.
Internal HGB early stopping is disabled to avoid random validation leaking duplicate groups.

Operating-regime exclusion refits remove entire clusters of oat/pa/ias from fitting and
calibration, then evaluate those regimes on final holdout. These diagnose operating-condition
shift. They do not establish unseen-engine accuracy because engine IDs were removed.
External datasets are used for descriptive shift diagnostics and predictions only.

The quick and full experiments use the same deterministic seed but different populations.
Do not repeatedly inspect final holdouts to tune configurations. Set the final configuration
using development evidence, freeze it, then run final evaluation. For extensive research,
add an immutable outer holdout plus grouped development CV before expanding the search.

## Model comparison

Raw features + Ridge margin baseline; raw/physics/statistical/combined features + histogram
gradient boosting; combined features + polynomial design-torque regression; and a reduced
feature ablation. Every candidate uses HGB fault classification and an OOF residual scale
model for heteroscedastic normal uncertainty. A predicted margin is available to the
classifier, using OOF values on fitting rows. This is why the raw baseline means raw measured
predictors plus the common, leakage-controlled stacking recipe.

Selection utility is 0.5*(ROC AUC - Brier) - 0.5*normal CRPS / train margin std. This is
an explicit internal utility balancing classification and probabilistic regression, not an
official challenge score. The chosen probability/distribution calibration is fit later on
independent calibration rows. To change the utility, edit phm/evaluate.py before the final run.

Feature selection uses selection-only permutation AUC ranking, keeps raw measurements,
and refits/evaluates the reduced model as another candidate. It only wins if overall utility
improves. Feature importances can mask correlated features; use ablations as the main evidence.
Training-only mutual information for both targets supplies another diagnostic.

## Confidence and uncertainty

- Calibrated p_faulty is used for probability reliability/Brier/ECE analysis.
- Conventional class labels use a calibration-selected F2 threshold and drive the confusion matrix.
- Submission class/class_conf maximize expected reward under the published asymmetric score;
  they can differ from conventional labels. class_conf is not p_faulty or the probability
  of the chosen class. Predictions CSVs preserve both outputs to avoid confusing them.
- Regression is exported as norm(loc=predicted_margin, scale=positive_sigma).
- Sigma is learned from grouped OOF absolute residuals and rescaled on calibration rows.
- NLL, CRPS, interval coverage/width, error bias and operating-condition slices are evaluated.
- Bootstrap intervals resample duplicate groups, not assets; unknown engine dependence limits them.
- The exact regression score normalization is not specified, so no official combined score is invented.

## Run outputs

| Output | Content |
|---|---|
| REPORT.md | Main readable results and limits |
| eda/ | Full-data summaries, correlations, distributions, class overlap, external shift |
| leaderboard.csv and candidates/ | Selection metrics for each candidate |
| permutation_importance_selection.csv | Feature ranking on selection rows |
| mutual_information_train.csv | Classification/regression MI on fitting rows |
| selected_spec.json | Frozen model/feature choice |
| selected_model.joblib | Exact evaluated model + calibration + config and fitting IDs |
| final_metrics.json | Internal final holdout classification/regression/uncertainty metrics |
| evaluation.png | Confusion matrix, reliability, predicted/actual margin, standardized residuals |
| bootstrap_intervals.json | Duplicate-group bootstrap metric intervals |
| conditional_performance.csv | Holdout metrics in training-defined operating bins |
| stress_tests.json | Regime-exclusion experiments or reason skipped |
| partition_manifest.csv | Original IDs, duplicate-group key and partition membership |
| config.json, input_manifest.json, environment.json | Config, input hashes and package versions |
| submissions/submission.jso | Test predictions in challenge format |
| submissions/validation_submission.jso | Validation predictions in challenge format |
| submissions/*_predictions.csv | Probabilities, labels, margin PDF parameters and intervals |
| completion.json | Proof the whole experiment completed and actual sample size |

## Current verification

See runs/verified_sample/REPORT.md for the supplied-data demonstration, if present in this
archive. Its experiment uses 20,000 rows; EDA uses all supplied rows. It is a harness validation
run, not the full-data final result. Execute configs/full.yaml for the full training experiment.
The conservative configuration is the primary demonstration, with altitude features disabled.
Ground-truth test/validation labels were not supplied, so no external accuracy is reported.
