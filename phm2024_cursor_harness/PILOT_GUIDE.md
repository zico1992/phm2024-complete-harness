# Current milestone

See [START_HERE.md](START_HERE.md) for the current commands. `shadow --labels` is now optional;
`evaluate` and `verify-evaluation` support delayed labels without model loading. `doctor` checks
runtime metadata before deserialization. The review queue shows priorities and latest statuses.
Python major.minor must match the artifact metadata; use Python 3.12 for the bundled sample.

# Frozen model review pilot

This extension wraps the saved JointModel/Calibrator without calling fit, changing features,
recalibrating probabilities, changing the classification threshold, or retraining the baseline.
Use it for engineering review and labelled shadow evaluation. It does not issue maintenance
commands or establish fitness for flight.

## Start the included demonstration in Cursor

Open this project folder. In PowerShell, activate your existing project environment.
The supplied sample artifact was saved with scikit-learn 1.8.0, NumPy 2.3.5 and pandas 2.2.3.
For the included demo, use requirements-pilot-demo.txt in a separate environment. For your
full baseline, use the environment in its environment.json; do not upgrade it to match the demo.

```powershell
py -3.12 -m venv .venv-pilot
.\.venv-pilot\Scripts\Activate.ps1
python -m pip install -r requirements-pilot-demo.txt
python -m unittest discover -s tests -v
python pilot.py serve --baseline pilot_artifacts/sample_demo --run-dir runs/pilot_replay_demo --port 8765
```

Open http://127.0.0.1:8765 in your browser. Keep the terminal running. Ctrl+C stops it.
The dashboard contains a searchable review queue, measurement details, fault probability,
margin interval, reliability reasons, labelled errors, validation slices, a measurement-entry
prediction interface and an append-only SQLite review history. Select an observation to populate
the measurement form. Reviewer names are self-reported. The SQLite database is local to the run.
A review disposition is not a confirmed diagnosis and never overwrites labels.

The included 2,999-row labelled replay uses the original 20,000-row SAMPLE model, not your
full selected model. It reproduces 17 FN and 14 FP. The additional 21,436-row test replay has
no labels and no measured accuracy. Challenge rows have no engine IDs/timestamps: no engine
history, flight alerts or fault lead time can be inferred from their row order.

## Register your actual frozen baseline

Use the original completed runs/full/selected_model.joblib, not a stress-test model or a refit.
Your full artifact was not supplied in this conversation; it is not included in this package.
Keep your existing data/ and runs/full/ directories when copying in this extension.

```powershell
python verify_run.py --run-dir runs/full --data-dir data
python pilot.py freeze --model runs/full/selected_model.joblib --training-csv data/X_train.csv --output pilot_artifacts/full_baseline --name full-selected-frozen
python pilot.py replay --baseline pilot_artifacts/full_baseline --input data/X_test.csv --output runs/full_pilot_replay
python verify_pilot.py --baseline pilot_artifacts/full_baseline --run-dir runs/full_pilot_replay --input data/X_test.csv
python pilot.py serve --baseline pilot_artifacts/full_baseline --run-dir runs/full_pilot_replay --port 8765
```

All output directories must be new/empty. Repeated runs use new names. Freeze copies the exact
model bytes and records SHA256. It fits ONLY a separate monitoring sidecar from the artifact's
fitting IDs, using calibration IDs for provisional monitoring thresholds. It never evaluates
the final holdout to choose those thresholds. If original runtime/source manifests are present,
their versions and training CSV hash are checked. Loading joblib requires a trusted source.

## What the reliability checks mean

| Check | Behaviour | Limitation |
|---|---|---|
| Missing/nonfinite measurements, impossible absolute temperatures, nonpositive torque/NG/NP, negative IAS, failed sensor quality | Row rejected, no prediction, review recommended | Data-quality screening, not an aircraft operating envelope |
| Outside fitting min/max | Names affected measurements | Inside each marginal range can still lack joint support |
| Joint seven-measurement and OAT/PA/IAS distances | Robust-scaled nearest neighbour against up to 50,000 unique fitting references; warning beyond calibration 99th percentile | Approximate empirical novelty, not probability of failure or validated OOD detection |
| Wide 90% margin interval | Warning beyond calibration 99th percentile width | Gaussian interval can be narrow and wrong under shift |
| Conditional coverage evidence | Warn if fewer than 100 calibration rows, no evidence, or observed 90% coverage below 85% | Exploratory evidence; no confidence bound or guaranteed operational coverage |
| Operational quality unknown | Review warning | Needs sensor-health feed |
| Exact fitting/calibration duplicate | Recorded and excluded from validation metrics | Does not establish independence of assets or engines |

Reliability thresholds are PROVISIONAL and recorded separately from the baseline. Changes need a
new sidecar/policy version and development validation. Never tune them to improve the inspected
holdout. Do not automatically reject regime 5: the ordinary full model had training coverage
there, while the exclusion stress test deliberately removed it. Model cluster IDs are specific
to their fitted artifacts and are not recovered engine identities.

`within_reference` means no implemented warning fired. It does not mean healthy, safe or
validated. A fault flag still recommends review. Every prediction stores calibrated p_faulty,
not the challenge's score-optimized class_conf. The fault flag uses the frozen F2 threshold,
which is not yet an approved operational maintenance threshold.

## Obtain the operational data

Use operational/templates/observations.csv, labels.csv, data_contract.json and
acceptance_criteria.json. See OPERATIONAL_DATA_REQUEST.md for owners and definitions.
No operational source connection or labelled operational observations were supplied; collection
and actual operational validation remain pending. Do not fabricate identifiers or timestamps
for the challenge CSVs and call them operational evidence.

Copy the templates to an operational_data/ directory. Populate:

- Observations: observation_id, engine_id, flight_id, timestamp_utc with explicit UTC offset,
  sensor_quality (`ok`, `verified`, `unknown`, `invalid`, `failed`) and all seven measurements.
- Outcomes: observation_id, faulty 0/1, trq_margin in percentage points, label_source and
  label_timestamp_utc with explicit UTC offset. A confirmed label must not precede its observation.
  Partial labels are allowed; unlabelled observations are counted separately.
- Contract: status `confirmed`, owner and confirmation date, exact baseline SHA256 from its
  manifest, exact baseline_physics, and confirmed meaning/source unit/baseline unit for every
  measurement. Scale/offset explicitly converts source values into the baseline units.
  Temperature baseline units must match C if that is the frozen configuration. Confirm PA/NP
  meanings and numerical scales with the dataset owner before operational use. Do not silently
  turn PA into altitude or NP into shaft power.

```powershell
python pilot.py intake --baseline pilot_artifacts/full_baseline --input operational_data/observations.csv --contract operational_data/data_contract.json
python pilot.py shadow --baseline pilot_artifacts/full_baseline --input operational_data/observations.csv --contract operational_data/data_contract.json --labels operational_data/labels.csv --output runs/shadow_cohort_001
python verify_pilot.py --baseline pilot_artifacts/full_baseline --run-dir runs/shadow_cohort_001 --input operational_data/observations.csv --contract operational_data/data_contract.json --labels operational_data/labels.csv
python pilot.py serve --baseline pilot_artifacts/full_baseline --run-dir runs/shadow_cohort_001 --port 8765
```

This command performs RETROSPECTIVE labelled shadow replay. It sorts real timestamps, saves
predictions before opening labels, then joins labels by observation ID. It does not simulate
real-time alert lead time, prove prospective availability or connect to a live telemetry stream.
For a prospective pilot, archive dated observation batches and frozen predictions before
outcomes become available, then evaluate matched outcomes in a separate immutable run.

Evaluation reports all eligible rows, including reliability warnings, and slices by regime,
reliability status and engine. Rejected/unlabelled/seen duplicate counts stay visible. Flight
metrics require fully labelled, valid, unseen-feature engine/flight groups. A flight alert means
any threshold crossing; persistence rules are not invented. No fault-onset lead time is reported
without event/onset timestamps. Challenge baseline engine independence remains unknown.
Acceptance is NOT ASSESSED until engineering agrees criteria and an independent operational
cohort provides evidence. New model comparisons require a separate development experiment.

## Local API and audit

GET /api/summary supplies run metadata, metrics and a local session token. GET /api/observations
supports q, filter (all/review/rejected/fn/fp), offset and a fixed 100-row page. GET /api/history
uses real engine identifiers only. GET /api/reviews returns the latest 200 review entries.
POST /api/predict accepts {"observations":[{"id":"example", ...seven raw measurements...}]},
1..100 rows, for a local measurement demonstration. POST /api/review accepts id, reviewer,
disposition (needs_inspection/monitor/data_issue/review_complete) and note. POST requests
require X-Session-Token from the current local session. Browser Origin and Host are checked.
The service binds to 127.0.0.1 only. Predict and review requests are logged to reviews.sqlite.

This is a single-user pilot, not a network deployment. There is no SSO, role authorization,
encrypted database, production queue, rollback orchestrator or aircraft-system integration.
Do not expose it on a network without adding the appropriate deployment controls. The Python
server is intentionally dependency-light and uses the existing model environment.

## Artifacts

Frozen baseline directory: manifest.json, baseline.joblib, reliability.joblib, policy.json.
Run directory: predictions.csv, optional evaluated_predictions.csv, evaluation.json,
run_manifest.json, completion.json, optional pilot_verification.json and reviews.sqlite.
Hashes cover inputs, labels, model, sidecar, policy and output prediction/evaluation files.
They detect accidental changes relative to the manifest; they are not cryptographic signatures
protecting against an attacker who can also rewrite the manifest.
