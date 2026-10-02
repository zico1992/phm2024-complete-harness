# PHM 2024 complete workspace: EDA through frozen-model review pilot

This ZIP contains the four original challenge CSVs, EDA, physics/statistical feature
engineering, selection, training, calibration, validation, regime investigation helpers,
frozen prediction/reliability tools, Windows cleanup fixes, runtime checks, delayed-label
snapshots and the prioritized review dashboard. Start with the path matching your situation.
Read CURSOR_TASK.md and AGENTS.md before asking Cursor Agent to change anything.

## A. You already completed my_full_run: resume without retraining

Extract the ZIP to a NEW folder. Do not overwrite your original experiment, data, model,
pilot baseline, existing prediction runs or review databases. Open the new project folder in
Cursor. Use your original full-model environment, or an environment matching its versions.
Your full model was saved with scikit-learn 1.9.1; the sample demo uses 1.8.0.
Use the Python major.minor recorded in the full run as well as its recorded package versions.
Do not reinstall the sample requirements in the full-model environment.

The full artifact is on your computer and was not uploaded here. Your uploaded metrics are
included under reference_results/user_full_run and explicitly marked metrics-only. They
are not a full run directory and cannot be used with verify_run.py.

PowerShell, from the extracted project root:

```powershell
# You can set this to the Python executable of your matching .venv-pilot instead.
$FullPython = "C:\Users\user\PHM_2024_Data_Challenge\phm2024_cursor_harness\.venv\Scripts\python.exe"
$OriginalRun = "C:\Users\user\PHM_2024_Data_Challenge\phm2024_cursor_harness\runs\my_full_run"
$OriginalData = "C:\Users\user\PHM_2024_Data_Challenge\phm2024_cursor_harness\data"

& $FullPython pilot.py doctor --model "$OriginalRun\selected_model.joblib"
& $FullPython verify_run.py --run-dir $OriginalRun --data-dir $OriginalData
& $FullPython pilot.py freeze --model "$OriginalRun\selected_model.joblib" --training-csv "$OriginalData\X_train.csv" --output pilot_artifacts/full_baseline --name full-selected-frozen
& $FullPython pilot.py doctor --baseline pilot_artifacts/full_baseline
& $FullPython pilot.py replay --baseline pilot_artifacts/full_baseline --input "$OriginalData\X_test.csv" --output runs/full_pilot_replay
& $FullPython verify_pilot.py --baseline pilot_artifacts/full_baseline --run-dir runs/full_pilot_replay --input "$OriginalData\X_test.csv"
& $FullPython pilot.py serve --baseline pilot_artifacts/full_baseline --run-dir runs/full_pilot_replay --port 8765
```

Use your actual interpreter location. If you already have a frozen baseline/run in the old
pilot project, you can point --baseline and --run-dir to those absolute paths instead of
freezing or replaying again. Legacy frozen manifests are supported; doctor lists packages
whose versions were not originally recorded. New freezes record a fuller runtime manifest.
All output directories must be new/empty. Open http://127.0.0.1:8765 and keep the terminal running.

## B. Explore the supplied sample demonstration

Use Python 3.12 and the sample's pinned dependencies in a separate environment:

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\Activate.ps1
python -m pip install -r requirements-pilot-demo.txt
python -m unittest discover -s tests -v
python pilot.py doctor --baseline pilot_artifacts/sample_demo
python pilot.py serve --baseline pilot_artifacts/sample_demo --run-dir runs/pilot_replay_demo --port 8765
```

The bundled model uses 20,000 training-source rows, with 2,999 independent final holdout rows.
It is not your full selected model. External test replay has no labels. The delayed_label_demo
snapshot uses actual challenge holdout labels and the exact saved predictions; it does not
represent operational engine data. Sample integration tests explicitly skip when the runtime
is incompatible, rather than unpickling a sample model under the full-model environment.
Generic contract/runtime/priority tests still run. A skipped test is not a successful sample
integration test; use the pinned demo environment for all tests.

## C. Start a new development experiment from EDA

This path is for a new development experiment, not the user already preserving my_full_run.
The requirements-pilot-demo.txt environment can run the whole modelling harness and sample
checks. requirements.txt contains the original general bounds; model persistence always
requires the recorded training runtime for later serving.

```powershell
python run.py --stage eda --data-dir data --output-dir runs/new_eda
python run.py --config configs/default.yaml --data-dir data --output-dir runs/new_quick
python run.py --config configs/full.yaml --data-dir data --output-dir runs/new_full
python verify_run.py --run-dir runs/new_full --data-dir data
python pilot.py freeze --model runs/new_full/selected_model.joblib --training-csv data/X_train.csv --output pilot_artifacts/new_full_baseline --name new-full-frozen
```

Training compares raw/physics/statistical/combined variants, performs selection on development
rows and calibrates separately before evaluating the frozen selection. Exact feature duplicate
groups stay together; targets and row IDs are never inference features. Unit assumptions are
explicit. See README.md and FEATURE_CATALOG.md for modelling details. Training a new model
creates a new candidate: it never replaces the existing baseline automatically.

## D. Operational predictions now, confirmed labels later

Obtain real observations and a confirmed contract using OPERATIONAL_DATA_REQUEST.md and
operational/templates/. No operational source connection or observations were supplied.
Never manufacture engine IDs or timestamps for challenge rows and call them operational data.
Use cumulative confirmed label exports for cumulative evaluation; each export creates a new
snapshot, with no automatic merge of earlier labels or review notes.

```powershell
& $FullPython pilot.py intake --baseline pilot_artifacts/full_baseline --input operational_data/observations.csv --contract operational_data/data_contract.json
# Labels are optional: save and review the predictions first.
& $FullPython pilot.py shadow --baseline pilot_artifacts/full_baseline --input operational_data/observations.csv --contract operational_data/data_contract.json --output runs/shadow_predictions_001
& $FullPython verify_pilot.py --baseline pilot_artifacts/full_baseline --run-dir runs/shadow_predictions_001 --input operational_data/observations.csv --contract operational_data/data_contract.json
& $FullPython pilot.py serve --baseline pilot_artifacts/full_baseline --run-dir runs/shadow_predictions_001 --port 8765
# Stop the dashboard with Ctrl+C before starting another server on the same port.

# When a confirmed label export is available:
& $FullPython pilot.py evaluate --baseline pilot_artifacts/full_baseline --run-dir runs/shadow_predictions_001 --labels operational_data/labels.csv --output runs/shadow_evaluation_001
& $FullPython pilot.py verify-evaluation --baseline pilot_artifacts/full_baseline --run-dir runs/shadow_evaluation_001
& $FullPython pilot.py serve --baseline pilot_artifacts/full_baseline --run-dir runs/shadow_evaluation_001 --port 8765
```

Evaluate reads saved predictions only. It does not load the model, rerun prediction, recalibrate
or retrain. It copies prediction bytes into a new snapshot, retains its exact label export,
records hashes and computes metrics. Source predictions/run files remain unchanged. Snapshots
refer to their source relatively; keep the source and snapshots together when moving folders.
Review history remains with the original prediction run, including when viewing later label
snapshots. Verify-evaluation checks hashes and recomputes label metrics without model loading;
verify_pilot.py against the source inputs checks reproduction by the frozen model.

Confirmation timestamps are compared to prediction commit time for transparency. They do not
prove prospective availability, indicate fault onset or establish detection lead time. This is
CSV batch shadow operation, not a connected live telemetry stream. Ground truth must include
independent provenance; a dashboard disposition is never a diagnosis/label.

## E. Review priority and status

| Tier | Trigger | Meaning |
|---|---|---|
| P1 | Fault flag and reliability warning | Review signal and support together |
| P2 | Fault flag without warning, or rejected input | Engineering signal review or data correction |
| P3 | Support/uncertainty warning without fault flag | Review reliability |
| P4 | No implemented review trigger | Routine observation |

These are provisional ordering rules, not aircraft safety severity or validated maintenance
thresholds. Outcome labels do not determine priority. The queue sorts by tier, then descending
fault probability and observation ID. Use Open reviews, Not yet reviewed, All observations
or Completed reviews. needs_inspection, monitor and data_issue remain follow-up; review_complete
closes that observation's review. Latest reviewer/time/note are visible; historical entries
remain append-only. Completing a review does not change a fault flag, probabilities or labels.
Separate prediction batches do not automatically share dispositions; no asset/flight persistence
or cooldown rule has been invented.

Runtime preflight runs before model loading in freeze, verify_run.py, serving and prediction.
It checks sklearn/NumPy/pandas, any recorded SciPy/joblib/threadpoolctl, and Python major.minor
when available. Missing source metadata requires the actual original environment.json via
--environment; versions are not inferred by unpickling. New runs/freezes record fuller metadata.
The checker does not install, upgrade or downgrade packages automatically.

## Limits

The service is local/single-user with self-reported reviewer identity. No SSO, telemetry feed,
maintenance commands, automated acceptance decision or unseen-engine proof is implemented.
Reliability thresholds and queue priorities require development/operational validation.
Confirmed outcomes and measurement definitions remain pending. The cloud browser cannot
reach loopback for visual QA; HTTP/API behaviour and JavaScript syntax were checked.
