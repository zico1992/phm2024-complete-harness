# PHM 2024 Harness Architecture Decision Record

Version 1.0 | 2 October 2026 | Record ADR 001 | Technical record for review

The harness uses a modular Python application with file based experiment artifacts, a frozen prediction and monitoring bundle, a local HTTP review application and a SQLite review ledger. This design makes the complete workflow reproducible in Cursor without an external service dependency. It supports a local engineering pilot; production identity, live telemetry integration and independently validated operational release remain future work.

This record is for the development owner, validation reviewer, operational integration engineer and IT security reviewer. Implemented decisions describe current code. Provisional decisions describe current rules awaiting stakeholder agreement. Deferred decisions are requirements for a future deployment, not completed capabilities. No named approver, production deployment authorization or maintenance authority is implied.

## 1 System context and boundaries

Inputs are challenge CSVs or operational observations converted through an owner confirmed data contract. Training produces a selected joint model, calibration and evidence files. Freezing packages the model unchanged with a reliability sidecar and a policy manifest. Replay or shadow inference writes predictions first. Labels can arrive later and create evaluation snapshots. Reviewers inspect a local dashboard and append dispositions to an audit database.

The output is a probability, torque margin distribution, conventional fault flag, reliability reasons and review priority. An engineer decides any next action. The application does not issue aircraft commands, replace an authorized diagnostic process, estimate remaining useful life or send automatic alerts to external systems.

The bundled executable artifact is a sample model trained under scikit learn 1.8.0. Uploaded full run results are reference evidence; the full 1.9.1 model remains in the user's original my_full_run directory. Registration must use that actual artifact and its matching environment. The sample must not silently become the user's full baseline.

## 2 Component architecture

| Component | Main files | Responsibility |
|---|---|---|
| Experiment driver | run.py and configs | EDA, selection, final fitting and artifact outputs |
| Statistical library | phm/data.py features.py models.py evaluate.py eda.py | Data contracts, features, joint model and metrics |
| Diagnostics | investigate_regime.py and verify_investigation.py | Regime support and prediction error investigation |
| Pilot entry point | pilot.py | Doctor, freeze, replay, shadow, evaluate and serve commands |
| Prediction and intake | operational/core.py | Frozen bundle, validation, reliability and cohort evaluation |
| Environment gate | operational/runtime.py | JSON preflight and trusted artifact loading |
| Delayed outcomes | operational/delayed.py | Immutable saved prediction evaluation snapshots |
| Review state | operational/review.py | Cohort scoped review history and priority policy |
| Local web app | operational/server.py and operational/web | HTTP endpoints and static dashboard |
| Verification | verify_run.py verify_pilot.py and tests | Hash, split, reproduction and workflow contracts |

Training and serving share the same feature and calibration implementation. There is no parallel reimplementation of the model in JavaScript. Dashboard JavaScript renders records and sends bounded API requests; Python owns predictions, validation, priority and audit state.

## 3 Artifact and data lifecycle

A modelling run uses a new output directory and records input hashes, effective configuration, environment metadata, partition IDs and duplicate groups. Candidate metrics and selection artifacts explain why the winning configuration was chosen. The final artifact includes its model, calibrator and fitting metadata. Holdout predictions and external submissions are produced from that saved selection. completion.json belongs to the actual completed directory; a guessed runs/full path is not interchangeable with my_full_run.

A frozen pilot baseline contains the original serialized model, a monitoring sidecar, policy and manifest. Hashes bind the three payload files; runtime metadata records supported dependencies. Freezing fits monitoring reference structures on allowed baseline fitting measurements and uses calibration rows for thresholds. It does not refit or recalibrate the predictive model or change its conventional threshold.

Each replay or shadow run retains predictions, input and baseline provenance, status summaries, any evaluation outputs and a manifest. Delayed evaluation copies the exact prediction bytes into a new snapshot, retains the label export and hashes, and computes metrics without loading the model. Source manifests and relative ancestry support shared review history. Keep recorded source directories available if shared history is needed.

## 4 Architecture decision register

### AD 01 Use a modular local application

Status: Implemented. Context: The user needs one harness covering research and a demonstrable review workflow in Cursor. Decision: Use Python modules and CLI orchestration, standard library HTTP serving, static HTML and JavaScript, and SQLite. Alternatives: Separate services, FastAPI plus a framework dashboard, or a hosted platform. Rationale: Low setup burden and traceable local execution suit the current evidence stage. Consequences: No production scaling, distributed jobs or service availability guarantees. Revisit when real integration, concurrent users or deployment requirements are confirmed. Evidence: run.py, pilot.py and operational/server.py.

### AD 02 Preserve one inference implementation

Status: Implemented. Decision: Store and reuse the fitted JointModel and Calibrator; invoke the same inference feature code for holdout verification and the pilot. Alternatives: Rebuild calculations in the web app or export a separate approximation. Rationale: One implementation limits training and serving drift. Consequences: Python serialization and dependency compatibility become deployment constraints. Revisit if an alternative runtime is needed and can reproduce predictions under an explicit tolerance. Evidence: phm/models.py, operational/core.py and verify_pilot.py.

### AD 03 Freeze the predictive baseline separately from monitoring

Status: Implemented. Decision: Package the unchanged predictive artifact with a separate support reference sidecar and policy. Alternatives: Retrain during pilot setup or embed undocumented serving adaptations. Rationale: The pilot must assess a stable baseline while collecting missing data. Consequences: New reference policies or model changes require a new version and evidence; warnings do not alter probabilities or flags. Revisit through a reviewed candidate release. Evidence: pilot.py freeze, operational/core.py and pilot_artifacts/sample_runtime_checked.

### AD 04 Reject incompatible runtime metadata before loading

Status: Implemented. Decision: Doctor and loader preflight use JSON metadata before deserialization. Core sklearn, NumPy and pandas versions must match exactly; recorded SciPy, joblib and threadpoolctl versions also match exactly. Recorded Python major and minor must match; patch differences are allowed. Alternatives: Ignore pickle warnings, automatically upgrade dependencies, or load first to infer versions. Rationale: Successful loading is not proof of cross version statistical equivalence. Consequences: A full 1.9.1 artifact needs its original compatible environment, while the bundled sample uses 1.8.0. Missing runtime metadata fails; legacy manifests disclose unrecorded auxiliary versions. A deserialization version warning is still treated as an error. Revisit only with a deliberate migration and prediction equivalence assessment. Evidence: operational/runtime.py and tests/test_runtime.py.

### AD 05 Trust only approved serialized artifacts

Status: Implemented boundary, limited integrity. Decision: Load user controlled trusted joblib artifacts after manifest and environment checks. Alternatives: Accept arbitrary uploaded models or implement a safer portable model format. Rationale: joblib preserves the fitted Python objects but may execute code during loading. Consequences: Hashes detect payload changes relative to a manifest; they are not signatures, and an attacker able to replace both manifest and payload can defeat them. The current app is not an untrusted artifact ingestion service. Revisit with signed releases, protected provenance and a reviewed serialization approach for production. Evidence: operational/runtime.py and operational/core.py.

### AD 06 Treat schema and units as an intake adapter

Status: Implemented, owner confirmation pending. Decision: Keep seven model inputs fixed and normalize operational units through a confirmed data contract. Require observation, engine and flight identifiers, timezone aware timestamps and quality fields. Reject placeholders, invalid conversions, invalid IDs and contradictory timing. Alternatives: Infer units from values or admit arbitrary telemetry fields into the model. Rationale: Model validity depends on semantic compatibility as well as column names. Consequences: An owner confirmation is an attestation, not automatic proof of sensor equivalence. NP and PA meanings must be resolved by the data owner. Revisit on sensor, unit, sampling or source system changes. Evidence: operational/templates/data_contract.json and operational/core.py.

### AD 07 Validate rows and retain their rejection evidence

Status: Implemented. Decision: Reject invalid rows without numerical predictions; retain reasons and counts. Checks include nonfinite or missing measurements, nonpositive torque or speeds, negative IAS, impossible absolute temperatures when Celsius is configured, failed quality and applicable altitude limits. Alternatives: Drop rows silently or impute at inference. Rationale: Reviewers need to see data failures and total intake coverage. Consequences: These screens are not a certified engine operating envelope. The positive NP gate is provisional given unresolved semantics. Warnings preserve model outputs but recommend review. Revisit with confirmed sensor tolerances and an approved missing data policy. Evidence: operational/core.py and tests/test_pilot.py.

### AD 08 Use reference based monitoring

Status: Implemented with provisional thresholds. Decision: Store marginal ranges and robust scaled nearest neighbour structures for seven raw inputs and three environmental inputs. The default reference is capped at 50000 unique fitting measurements, seeded reproducibly; calibration sets 99th percentile distances and width limits. Regime coverage adds a calibration based warning. Alternatives: Use only marginal ranges or learn a separate production error classifier. Rationale: Joint novelty can exist inside every individual feature range. Consequences: Reference sampling may miss rare conditions; metrics and thresholds are heuristic support measures. They do not certify extrapolation or coverage. Revisit with observed operational errors, reviewer load and reference population growth. Evidence: operational/core.py and tests/test_pilot.py.

### AD 09 Write predictions before evaluating outcomes

Status: Implemented. Decision: Persist predictions and a commit timestamp before optional labels are joined. Delayed evaluation uses the saved file, not a new inference request. Alternatives: Rerun a changed model when labels arrive or overwrite a prior result. Rationale: Historical predictions must remain assessable against later ground truth. Consequences: Stored timestamps and hashes support traceability but do not independently prove prospective collection. Evaluation works without model loading; later ground truth cannot affect prediction or queue ordering. Revisit for a tamper resistant acquisition system. Evidence: operational/delayed.py and tests/test_delayed.py.

### AD 10 Version label evaluations as immutable snapshots

Status: Implemented. Decision: Write each evaluation to a fresh directory with labels, exact prediction bytes, metrics, provenance and hashes. Validate unique label IDs, binary finite outcomes and confirmation timing. Alternatives: Mutate the source run or maintain an implicit incremental label store. Rationale: Partial exports and corrected labels need reviewable history. Consequences: Labels are not automatically accumulated; cumulative assessment requires a cumulative export. A corrected label creates a new version. Source ancestry is required for shared review history, and changed, cyclic or missing ancestry is rejected. Revisit with an operational label registry and adjudication process. Evidence: operational/delayed.py, operational/review.py and verify_pilot.py.

### AD 11 Scope reviews to the prediction cohort

Status: Implemented. Decision: Key review history by baseline SHA and prediction CSV SHA, with observation ID inside that scope. Resolve delayed snapshots to the original run's database and derive current state from the latest audit sequence. Alternatives: Use observation ID alone or duplicate independent review databases for each label snapshot. Rationale: Reused IDs must not contaminate different model or batch histories. Consequences: Different cohorts do not automatically share dispositions even when IDs match. Latest review_complete closes the open queue without altering predictions. Application history is append only; local file access can still modify the database. Revisit with a central identity and audited storage design. Evidence: operational/review.py and operational/server.py.

### AD 12 Make queue rules explicit and reversible

Status: Provisional implemented ordering. Decision: P1 flagged faults with support warnings, P2 supported flagged faults or rejected data, P3 warnings without flags, P4 routine. Ties use descending probability then ID. Review states are unreviewed, follow_up and complete. Alternatives: A severity score, automatic maintenance disposition or label driven priority. Rationale: The current pilot needs clear human review ordering. Consequences: The rules do not establish physical severity, urgency, persistence or business value. Queue completion does not change the underlying model assessment. Revisit after engineering owners approve actions and measure capacity. Evidence: operational/review.py and operational/web/app.js.

### AD 13 Restrict the web interface to loopback

Status: Implemented local boundary. Decision: Bind to 127.0.0.1, validate Host, require a session token and matching or absent Origin for POSTs, omit CORS and serve a self restricted content security policy. Bound request size to 1 MB and prediction requests to 100 rows. Render record text using textContent. Alternatives: Expose a network port or add production identity infrastructure now. Rationale: A local demonstration should minimize unnecessary network exposure. Consequences: This is not authentication or authorization for a shared deployment. Review identity is self reported; no SSO, RBAC, TLS termination or protected audit identity is implemented. Revisit before any remote or multiuser deployment. Evidence: operational/server.py and tests/test_pilot_workflow.py.

### AD 14 Close database and server resources explicitly

Status: Implemented; Windows execution pending locally. Decision: Wrap database transactions with explicit connection closure, and close HTTP server sockets through context management. Alternatives: Rely on garbage collection or ignore temporary directory cleanup errors. Rationale: SQLite transaction context management does not itself close a connection, which can leave reviews.sqlite locked on Windows. Consequences: Close and rollback regression tests cover the mechanism; current execution evidence is Linux, so the user's Windows suite must confirm the platform outcome. Revisit if concurrent serving or shutdown behaviour changes. Evidence: operational/server.py and workflow and database contract tests.

### AD 15 Separate candidate training from promotion

Status: Implemented separation; approval workflow deferred. Decision: Train into new directories, verify artifacts, then explicitly freeze a chosen candidate. Existing baselines and runs are preserved. Alternatives: Overwrite a baseline or update online. Rationale: Stable operational evidence requires versioned candidates. Consequences: No automatic promotion, online learning or rollback controller exists. Retaining the old frozen bundle enables manual rollback. Revisit when a release owner and deployment process are assigned. Evidence: run.py, pilot.py, CURSOR_TASK.md and AGENTS.md.

## 5 Interface and evaluation contracts

The CLI exposes development, verification, baseline registration, replay, shadow intake, delayed evaluation and local serving stages. Each stage uses the actual artifact directory and records its dependencies. The API provides summaries, observations, observation details, recent scoped reviews, history, review submission and bounded prediction requests. Operational intake through the CLI carries metadata; the local prediction endpoint is a measurement demonstration with the seven predictors and is not a live telemetry integration contract.

The dashboard displays probability, margin mean and scale, intervals, reliability explanations, priority, review state and available labelled errors. Open, completed and unreviewed filtering separates workload from the retained record. Known errors may be inspected diagnostically but do not change review priority. No review disposition changes predictions or calibration.

Verification checks data hashes, duplicate group and partition separation, holdout reproduction, submission coverage and schema, baseline payload integrity and replay prediction reproduction. Delayed verification recomputes label metrics and verifies snapshot hashes without deserializing a model. Detached snapshot metric validation and source ancestry validation for shared reviews are distinct checks.

## 6 Quality attributes and limitations

| Attribute | Current mechanism | Remaining limitation |
|---|---|---|
| Reproducibility | Seeded configuration, hashes, environment record and saved predictions | No universal cross version reproducibility |
| Statistical isolation | Group partitions, OOF features and independent final labels | Hidden engine dependence and selection reuse |
| Traceability | Run manifests, snapshot provenance and scoped SQLite history | No signed manifests or tamper resistant ledger |
| Availability | Local files and one HTTP application | No high availability or recovery service |
| Security | Loopback, bounded requests and session checks | No shared user identity or production authorization |
| Scalability | Bounded batches and sampled neighbour reference | No distributed inference or load test evidence |
| Maintainability | Separate modelling, runtime, intake, review and UI modules | Deployment and migration ownership unassigned |
| Operability | Doctor, verifiers, tests and documented commands | No telemetry scheduler, service health platform or alert transport |

## 7 Verification evidence

The current milestone records 34 passing tests in Python 3.12 and scikit learn 1.8.0. They cover data and score contracts, frozen prediction equivalence, artifact tampering, runtime mismatches before loading, invalid row rejection, joint novelty within marginal bounds, delayed labels and review scope, resource closure and local API boundaries. The bundled 20000 row modelling run, 2999 row labelled replay, 21436 row unlabelled replay and 2999 row delayed evaluation verify independently.

A refreshed sample manifest has the same predictive artifact, monitoring sidecar and policy hashes as the earlier sample bundle; it adds full runtime metadata without baseline training. The full user artifact was not executed here. No labelled operational cohort was supplied. Temporary operational style fixtures test contracts and are not operational evidence. Dashboard JavaScript syntax and API behaviour were checked; visual loopback browser verification was blocked. Production load and Windows execution evidence remain pending.

## 8 Next architecture milestone and release conditions

First obtain an owner confirmed data contract and labelled operational observations. Use the frozen baseline in shadow mode and preserve prediction before label evidence. Agree review actions and acceptance criteria before evaluating the independent cohort. Keep the reference data request and null acceptance template visible until the respective owners approve them.

Before live integration, specify event IDs and ordering, sensor sampling and unit adapters, retry and idempotency, label corrections, retention, operational ownership and reviewer capacity. Before shared deployment, add authenticated identity, role based authorization, protected transport, controlled artifact distribution, backups and recovery, audit protection and observability. These requirements are deferred and no infrastructure choice is committed by this record.

If a retrained candidate is considered, compare it with the frozen baseline on independent observations and assess both predictive quality and review burden. Record the chosen model hash, environment, data contract, policy and acceptance decision. Release should be explicit and reversible, preserving prior evidence. The system currently offers a local review pilot rather than an automatic deployment gate.

## 9 Evidence index and decision maintenance

Primary evidence: run.py; pilot.py; phm/data.py; phm/features.py; phm/models.py; phm/evaluate.py; operational/core.py; operational/runtime.py; operational/delayed.py; operational/review.py; operational/server.py; operational/web/app.js; verify_run.py; verify_pilot.py; tests; MILESTONE_REPORT.md; BUNDLE_STATUS.json; START_HERE.md; CURSOR_TASK.md; AGENTS.md; operational/templates/data_contract.json and acceptance_criteria.json.

A companion evidence manifest records SHA256 digests for implementation and reference files at this record's creation. It identifies the reviewed snapshot; it does not sign or authenticate the files. Changes to units, serialization, splitting, calibration, support policy, label eligibility, cohort scope or exposure boundary should create a revision referencing the affected AD and SD identifiers. Proposed responsibility: development owner for architecture, data owner for intake, engineering lead for review policy, IT security for deployment boundary and validation reviewer for independent evidence. Assign actual approvers and dates before formal adoption.
