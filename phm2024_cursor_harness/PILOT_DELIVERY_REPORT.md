# Prediction and review pilot delivery

Implemented: frozen prediction wrapper; separate monitoring sidecar; strict operational
intake; CSV replay; retrospective labelled shadow evaluator; loopback JSON API; local review
dashboard; SQLite review/prediction audit; run/artifact hashes and prediction reproduction.
Updated CURSOR_TASK.md and AGENTS.md. Historical modelling task retained separately.

## Actual completed evidence

| Check | Outcome |
|---|---|
| Baseline identity | Included 20,000-row sample artifact, unchanged SHA256 |
| Labelled challenge holdout replay | 2,999 observations; all predictions reproduced |
| Classification replay | FN 17, FP 14, fault recall 98.604%, accuracy 98.966% |
| Margin replay | MAE 0.7976 percentage points; 90% interval coverage 91.63% |
| Labelled replay review workload | 1,274 recommendations: fault flags and/or reliability warning |
| External test replay | All 21,436 observations; labels absent, no measured accuracy |
| External sample-model support warnings | 9,159 rows (~42.73%) have joint/environment coverage-gap warning; thresholds provisional |
| Contract tests | 21 passed, including end-to-end shadow contract fixture and local API/review boundary tests |
| Output verification | Both replay prediction sets reproduced and hashes checked |
| Dashboard JavaScript | Syntax checked; local HTTP assets and endpoints checked |
| Visual browser validation | Not completed: cloud browser blocked the loopback URL |
| Labelled operational evaluation | NOT RUN: no operational observations or confirmed contract supplied |
| Full baseline registration | Pending: user full-run selected_model.joblib is not present here |

Baseline SHA256: `658812bb98fbc71aec0ca7148010d3ce9dbca9f83229054a474c349e5e79aa51`.
This SHA identifies the included SAMPLE model. It is not the full-run baseline's identity.
No baseline/model fit or probability/threshold recalibration occurred in this extension.

Reliability checks flag invalid data, sparse joint support, out-of-range measurements,
wide predicted intervals, insufficient/weak conditional calibration evidence and unknown
operational quality. They do not automatically reject a regime. Flags are review context,
not approved flight/maintenance decisions or guarantees of coverage. The sample's external
support warnings must not be presented as measured failures or full-model warning rates.

The temporary shadow workflow test assigns explicitly synthetic metadata to challenge rows
inside a test directory to exercise the plumbing. Its output is not shipped as an operational
run and is not evidence of unseen-engine performance.

## Next required handoff

1. Open PILOT_GUIDE.md to run the bundled demo in Cursor.
2. Register the original full-run artifact in its original environment; preserve the champion.
3. Obtain the confirmed measurement dictionary and real engine/flight/time/quality data plus
   independently confirmed labels using OPERATIONAL_DATA_REQUEST.md and the templates.
4. Validate intake, run shadow, verify outputs, inspect the dashboard and agree operational
   acceptance against an independent cohort. Actual operational collection remains pending.

The service is local and single-user; reviewer identity is self-reported. There is no telemetry
connector, online scheduler, SSO or network deployment. Fault onset, lead time, persistence and
remaining useful life are not inferred. Exact training/calibration feature duplicates and
invalid rows are excluded from evaluation with counts visible; engine independence is unknown.
