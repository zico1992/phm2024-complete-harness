# Cursor task: frozen-baseline maintenance review pilot

Read START_HERE.md, AGENTS.md, PILOT_GUIDE.md, OPERATIONAL_DATA_REQUEST.md and FEATURE_CATALOG.md.
The modelling phase is complete. Keep the current selected model frozen. Implement and
maintain the prediction interface, reliability checks and review workflow; do not rerun
training or select a new champion as part of this task.

1. Inspect the existing runs/my_full_run artifacts and original environment.json. Run verify_run.py
   against the original data. The full selected_model.joblib was not uploaded to ChatGPT;
   locate it locally. Do not replace it with the included sample model or regime stress models.
2. Use the full model's original environment. Test the included sample demonstration in a
   separate pinned environment if versions differ. Run the contract tests; no baseline fit.
3. Freeze the exact full artifact with pilot.py freeze (PILOT_GUIDE.md commands). Preserve the
   original file, record SHA256 and create a separate reliability sidecar using only fitting
   and calibration IDs. No final-holdout threshold tuning. Save to a new directory.
4. Replay existing test/validation CSVs using pilot.py replay. Verify with verify_pilot.py.
   These CSVs have no labels: do not report accuracy. The included labelled 2,999-row demo
   uses the SAMPLE model and is harness evidence only. Never manufacture time/asset IDs.
5. Launch pilot.py serve and verify the review queue, search/filter/pagination, observation
   detail, manual prediction, validation evidence and append-only review notes. Keep the
   service local to 127.0.0.1. Labels and review dispositions must remain separate.
6. Complete the operational intake request with the data owner. Populate the templates with
   real engine/flight/time/quality metadata and independently confirmed outcomes. Confirm
   PA/NP definitions and all units, conversions and baseline SHA. Do not send external
   messages unless explicitly authorised by the user. Report missing inputs accurately.
7. Validate operational CSVs with pilot.py intake. Agree acceptance criteria before using an
   independent cohort for decisions. The supplied criteria template is unapproved and the
   code reports acceptance NOT ASSESSED; no invented pass/fail threshold.
8. Execute pilot.py shadow on the labelled cohort. Predictions must be saved before labels
   are opened. Run verify_pilot.py, then launch the same dashboard on that shadow run.
   Report eligible/invalid/unlabelled/seen-duplicate counts, FN/FP, calibration, margin
   error/coverage, condition/engine slices and complete-flight false alerts/missed faults.
9. Distinguish retrospective labelled replay from prospective shadow operation. No lead-time
   claim without actual fault onset events. No unseen-engine claim without asset identity
   evidence. No row-order history, automatic maintenance action or flight-control integration.
10. Record observations, reviews and inspection follow-up. Confirm the baseline hash is
    unchanged at completion. Any model change must be a separate versioned development
    experiment and must not overwrite this champion or operational evidence.

Deliver completion.json, pilot_verification.json, evaluation.json and a concise pilot report.
If operational observations are missing, complete the replay/UI/intake work and explicitly
mark operational evaluation pending. The earlier modelling task is archived in
MODELLING_TASK_ARCHIVE.md; do not execute it during frozen-baseline pilot work.

## Current milestone additions
- Run pilot.py doctor against the actual full model or frozen baseline before model loading.
  Match original runtime; sample and full environments are separate. New metadata records
  auxiliary versions. Missing legacy versions are reported rather than invented.
- Shadow predictions may precede labels. Save the immutable source run, evaluate each confirmed
  cumulative export with pilot.py evaluate in a new directory, then verify-evaluation. Never
  recompute predictions, overwrite source runs or use review dispositions as ground truth.
- Use provisional P1/P2/P3/P4 queue ordering based only on prediction/reliability signals.
  Show latest review status; review_complete removes it from open reviews, not from audit.
  Delayed snapshots reuse source prediction review history. Keep snapshot ancestry intact.
- Verify runtime failure happens before deserialization, delayed evaluation never loads the
  model, hashes/source predictions are immutable, and review status persists across snapshots.
- START_HERE.md has complete EDA-to-pilot commands. The user already has my_full_run; do not
  retrain it while updating the pilot. Bundled full results are metrics-only, not a saved model.
