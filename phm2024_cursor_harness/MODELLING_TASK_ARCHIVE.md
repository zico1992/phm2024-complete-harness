# Paste this task into Cursor Agent

Work through the PHM 2024 project in this folder. Follow AGENTS.md and preserve the
existing leakage controls. First read README.md and FEATURE_CATALOG.md.

1. Verify Python and install requirements in an isolated environment. Run python -m unittest discover -s tests -v.
2. Run EDA on the supplied CSVs. Summarize duplicate groups, label overlap, shift and
   uncertain units. Do not interpret id as time or assume faulty iff torque margin<0.
3. Run the quick config. Verify completion.json, partition_manifest.csv and all outputs.
4. Inspect selection-partition ablations and feature ranking. Discuss physical plausibility
   and redundant/correlated features. Change candidate grids using only development data.
5. Ask the dataset owner or locate primary metadata for PA/NP and measurement units.
   Until confirmed, keep conservative config as the primary analysis. The altitude config
   is a labelled hypothesis. If testing it, compare selection results only, and do not
   use final holdout results to choose semantics or features.
6. Before a final experiment, fix features, candidate grid, thresholds/calibration recipe,
   seeds and selection utility. Run configs/full.yaml on every training row. Once
   selected_spec.json is frozen, use final holdout exactly once for the final analysis.
7. Read final_metrics.json and stress_tests.json. Explain FN risk, performance by operating
   condition, calibration and interval coverage. Distinguish internal holdout from truly
   unseen engines. Do not claim external accuracy without y_test/y_validation.
8. Validate both .jso submissions against their corresponding original IDs. Check every
   scale>0, finite values, legal classes/confidences, correct row counts and norm arguments.
   Run python verify_run.py --run-dir runs/full --data-dir data to check these contracts.
9. Deliver a concise summary linking REPORT.md, selected_model.joblib, feature ranking,
   uncertainty results and submission files. State sample size, partitions and limitations.

Optional extensions, only after the base run is complete:
- Add LightGBM or CatBoost models using the same splitting/OOF/calibration contracts.
- Compare identity-link positive-clipped polynomial design-torque regression with the
  existing log-link variant; tune degree/regularization only on development data.
- Add grouped cross-validation across development seeds, with an untouched outer holdout.
- Add conditional quantile regression, conformal intervals, or an ensemble variance model.
  Proper PDF export still requires an explicit family and positive scale.
- Estimate density-ratio importance weights from training vs external features as a separate
  transductive experiment; identify it explicitly. Do not mix with inductive baseline results.
- Explore design-torque consistency conditional on oat/pa/ias. Target reconstruction is
  training-only supervision, never a test-time feature.

Stop and report a genuine unresolved data-contract violation. Otherwise finish the full
authorized workflow. Keep every completed run so results can be reproduced and reviewed.
