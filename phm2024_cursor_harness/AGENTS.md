# PHM 2024 modelling contract

Read README.md, FEATURE_CATALOG.md, and CURSOR_TASK.md before changing the harness.
The user wants EDA, physics/derived/statistical features, feature selection, model
training, validation and final model analysis. Execute the work and record evidence.

## Data contracts
- Join X_train and y_train on id with one-to-one validation. Preserve original row IDs.
- Never use id as a predictor or treat row order as time. There are no reliable asset IDs.
- Keep exact duplicate feature groups together in every split, including auxiliary OOF folds.
- Never add target-derived design torque or actual trq_margin to inference features.
- Predicted-margin stacking must use fold-trained models on fitting rows.
- Fit ECDF, scalers, clusters, conditional statistics, imputers and feature selectors only on fitting partitions.
- Temperature and altitude/airspeed unit assumptions must appear in config and reports.
- PA is ambiguous between official and paper descriptions. No silent altitude interpretation.
- NP is ambiguous. Do not call torque*np measured shaft power or efficiency.
- No genuine thermal efficiency, pressure ratio of the compressor, mass flow or remaining useful life is observable here.
- Test and validation CSVs have no labels. They provide predictions and shift diagnostics, never measured accuracy.

## Experiment contracts
- Compare raw, physics, statistical and combined ablations on identical duplicate-group splits.
- Select candidates/features on selection data only. Calibrate on a separate calibration partition.
- Freeze selected_spec.json before final holdout evaluation. No retuning after seeing holdout results.
- Stress-test operating regimes as proxies, not recovered engines. Do not drop np<ng samples by default.
- Report accuracy, fault recall, precision, F1/F2, FN/FP, AUC, Brier, ECE, MAE/RMSE/R2, NLL/CRPS and interval coverage.
- Reproduce the published classification formula. Do not claim an official combined score without exact regression normalization.
- Preserve calibrated p_faulty separately from score-optimized class_conf.
- Normal distribution scales must remain finite and positive.
- Save the exact evaluated artifact. A full-data refit needs fresh calibration and is a different model.
- New variants require a new output directory, config, seed, data hashes and selection ledger.
- Tests must verify leakage boundaries and meaningful mathematical contracts.
- Never report results as run unless completion.json and the referenced metrics exist.


## Active phase: frozen-baseline operational review pilot
- Read PILOT_GUIDE.md and OPERATIONAL_DATA_REQUEST.md. CURSOR_TASK.md now directs pilot work;
  MODELLING_TASK_ARCHIVE.md is historical, not an instruction to retrain the champion.
- Never call fit, alter selected features, recalibrate or change the current baseline threshold
  in pilot code. Preserve the original selected_model.joblib and its SHA256.
- The full user-run artifact is not bundled. The included sample artifact is demonstration
  only. Never substitute sample or regime-exclusion artifacts for the full baseline.
- Monitoring sidecars are separate artifacts fitted only on saved fitting IDs; monitoring
  thresholds/evidence use calibration IDs, never inspected final holdout or external labels.
- Reliability checks are provisional; within_reference is not a safety certificate. No regime
  is automatically rejected. Cluster numbering is artifact-specific, not an engine identity.
- Use real engine IDs, flight IDs and explicit-offset timestamps for operational history.
  Challenge row IDs/order cannot supply chronology. Do not fabricate operational observations.
- Require a confirmed measurement contract, explicit conversion and confirmed label provenance.
  Record missing operational access/data; do not claim collection or operational evaluation ran.
- Save predictions before reading labels. Never use truth or review notes as inference features.
  Review dispositions remain separate from verified outcomes. Partial labels stay counted.
- Exclude exact fitting/calibration feature duplicates and rejected rows from measured metrics,
  while exposing their counts. Include reliability-warning rows in overall evaluation; do not
  hide difficult rows. Report slices and complete-flight denominators explicitly.
- Do not claim unseen-engine independence when challenge engine identities are unavailable.
  Flight alert proxies are any threshold crossing, not fault onset, persistence or lead time.
- No automated maintenance/flight actions, self-learning or public/network deployment in this
  pilot. Local review service binds to 127.0.0.1. New deployment scope is separate work.
- Every new run/sidecar needs a new directory and manifests. Verify prediction reproducibility,
  output hashes and baseline immutability with verify_pilot.py. Keep completed evidence.
- Stakeholder acceptance criteria remain unapproved until supplied; report NOT ASSESSED.
- Tests must cover invalid inputs, immutable baseline predictions, label separation, temporal
  contracts, duplicate exclusion, joint novelty and local prediction/review endpoints.

## Runtime, delayed labels and review queue contracts
- Check recorded runtime JSON before deserializing any persisted model; turn inconsistent
  estimator-version warnings into errors. Never silently bypass runtime checks.
- Use original environment metadata; do not infer source versions by unpickling. Legacy
  unrecorded dependency versions must be explicit. Separate sample/full-model runtimes.
- Delayed evaluation operates on hashed saved predictions and confirmed label exports only.
  No model loading, inference, recalibration or fit. Write a new evaluation snapshot; never
  edit source run files. Supply cumulative exports for cumulative coverage.
- Preserve relative snapshot ancestry and scope review history by frozen baseline and
  prediction hash. Completed reviews remain audit entries; labels and notes remain separate.
- Queue priority uses predictions and reliability checks, never actual labels or residuals.
  All priority rules are provisional; they do not establish fault severity or flight safety.
- close every SQLite connection explicitly after transaction handling. Never suppress Windows
  cleanup failures. Small-slice undefined metrics are null rather than hidden warnings.
