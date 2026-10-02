# PHM 2024 Harness Statistical Design and Decision Record

Version 1.0 | 2 October 2026 | Record SDDR 001 | Technical record for review

This record defines how the harness develops, selects, evaluates and monitors a joint fault classifier and torque margin probability model. Its present intended use is engineering review in a local replay and operational shadow pilot. The internal results support further evaluation; they do not establish unseen engine performance or authorize maintenance decisions. The deployed baseline remains frozen while operational measurements and delayed labels are collected.

The statistical audience is the modelling owner, independent validation reviewer, operational data owner and maintenance engineering lead. No formal owner or approver is assigned by this record. Implemented decisions describe executable behaviour. Provisional decisions require operational agreement. Proposed changes below are a future plan and are not represented as implemented.

## 1 Scope and evidence boundary

The record covers EDA, feature construction, feature selection, joint model training, probability and scale calibration, holdout assessment, regime diagnostics, frozen reliability checks, delayed label evaluation and dashboard prioritization. It excludes remaining useful life, fault onset timing, automated aircraft control and certified maintenance release.

The bundled sample has a reproducible saved artifact and verification outputs. The full run reference contains uploaded metrics and predictions, but no executable full model or completion file. The full model is held in the user's original my_full_run directory and was trained with scikit learn 1.9.1. Its successful verification is user reported. Sample verification here used Python 3.12 and scikit learn 1.8.0. These evidence populations must remain separate.

| Evidence population | Extent | Permitted interpretation |
|---|---|---|
| Original labelled challenge training | 742625 rows | Development and internal group holdout |
| Bundled sample experiment | 20000 rows and 2999 final rows | Executable end to end demonstration |
| Uploaded full run holdout | 111381 rows | Reported internal performance reference |
| External challenge test | 21436 rows without labels | Predictions and support diagnostics only |
| Labelled operational cohort | Pending | Required for operational acceptance |

## 2 Data contract and statistical unit

The seven predictors are trq_measured, oat, mgt, pa, ias, np and ng. The id field joins records and is never a predictor. Training targets are binary faulty and continuous trq_margin in percentage points. Torque margin follows 100 times measured torque minus design torque, divided by design torque. It is not a substitute for the faulty label: both healthy negative margins and faulty nonnegative margins occur.

The challenge observations have no engine identifier or timestamp. Their shuffled rows cannot support temporal features, degradation trajectories or a verified engine independent split. An exact raw measurement fingerprint defines the duplicate group. Each group stays in one partition, although distinct observations from the same unknown engine may still be dependent.

Challenge loaders require the exact schema, unique nonnegative integer IDs, finite predictors, a one to one target join, binary fault labels and finite margins. Positive torque and margin greater than minus 100 permit positive design torque reconstruction. Missing values fail ingestion; no imputation policy is silently introduced. Operational intake separately requires confirmed meanings and units, engine and flight IDs, sensor quality and timezone aware observation timestamps. Unit conversion is explicit and recorded.

PA is opaque by default because its meaning and units are unresolved. NP interpretation also remains unresolved. Celsius temperature interpretation is explicit in the configuration. IAS units are unknown by default. These assumptions limit physical interpretation even when predictive performance is strong.

## 3 Statistical objectives and estimands

The classifier estimates a calibrated probability of the challenge fault label conditional on the seven measurements. A calibration selected threshold produces the conventional fault flag. The regressor estimates conditional mean torque margin and a positive, observation dependent Gaussian scale. Intervals describe that empirical distribution under conditions resembling the development data.

Primary internal quantities are faulty recall and precision, false negative and false positive counts, discrimination, probability calibration, margin MAE and RMSE, normal CRPS and empirical interval coverage. Operational quantities must additionally include missed fault positive flights, false alerts per 100 healthy flights, rejected input rate, label completeness and performance by engine, flight and operating conditions. A row level probability is not a fault severity, maintenance urgency or causal diagnosis.

## 4 Partition design and fitting boundary

GroupShuffleSplit with seed 42 allocates approximately 60 percent to initial fitting, 15 percent to selection, 10 percent to calibration and 15 percent to final evaluation. Proportions are group based and row counts can differ from exact percentages. The final model is refitted on fitting plus selection rows; calibration stays separate and the final holdout remains excluded from all fitted components.

| Partition | Approximate share | Allowed use |
|---|---|---|
| Initial fitting | 60 percent | Learned features and candidate models |
| Selection | 15 percent | Candidate comparison and permutation ranking |
| Calibration | 10 percent | Probability calibration, scale correction and F2 threshold |
| Final holdout | 15 percent | Locked internal metrics and exploratory diagnostics |

Feature preprocessing is fitted inside each stacking fold. Neither final labels nor external challenge labels are available for fitting. Full data EDA is descriptive and precedes optional target independent sampling. Once final outcomes have been inspected for regime investigation, that holdout cannot serve as untouched evidence for subsequent tuning. A new independent cohort is then needed for confirmatory evaluation.

## 5 Feature and model design

Raw features provide the control candidate. Physics derived attributes include temperature normalization, corrected NG, thermal differences and ratios, NP to NG relationships, torque ratios and selected interactions. These are empirical physics informed proxies rather than thermodynamic efficiency or shaft power measurements. Optional atmosphere, density and airspeed features require explicit pressure altitude meaning, altitude units and IAS units. They remain disabled in the default configuration.

Statistical attributes include robust median and MAD deviations, empirical distribution positions, marginal tail indicators, and six operating clusters fitted on RobustScaler transformed OAT, PA and IAS. Distances, memberships and deviations from regime medians describe multivariate operating conditions. Cluster numbers have meaning only within a specific fitted artifact; regime 5 is not a universal operational category.

The full configuration compares six candidates: raw linear, raw direct, physics direct, statistical direct, combined direct and combined design torque reconstruction. Direct margin models use histogram gradient boosting. The linear margin control uses scaled Ridge regression. The reconstruction candidate learns log design torque from environmental predictors with degree two polynomial Ridge regression and then reconstructs margin. Every candidate's classifier is histogram gradient boosting; this is not a broad classifier family search.

The full boosting configuration uses 220 iterations, 31 maximum leaves, learning rate 0.08, regularization 1 and no early stopping. GroupKFold supplies five out of fold margin predictions for classifier stacking. The scale model learns log absolute out of fold residuals with a 0.05 floor, then applies a Gaussian conversion factor. Inference uses a regressor fitted on all fitting rows. This introduces a possible difference between stacking features during training and inference, which should be examined on new cohorts.

## 6 Statistical decision register

### SD 01 Group exact repeated measurements

Status: Implemented. Decision: Use measurement fingerprints as the grouping constraint in splits and stacking. Alternatives: Independent random rows or actual engine groups. Rationale: Random rows would place identical measurements in training and evaluation; engine groups are unavailable. Consequence: Duplicate leakage is reduced but hidden engine dependence remains. Revisit when engine identity and time become available. Evidence: phm/data.py and tests/test_contracts.py.

### SD 02 Gate physical interpretations

Status: Implemented with provisional semantics. Decision: Keep PA opaque and enable atmosphere calculations only through an explicit unit hypothesis. Alternatives: Infer altitude and speed units from ranges or omit all derived attributes. Rationale: A plausible range cannot confirm a sensor's meaning. Consequence: Useful proxy features remain available, but claims about physical mechanisms are constrained. Revisit after the operational data owner confirms every field. Evidence: configs/full.yaml, phm/features.py and FEATURE_CATALOG.md.

### SD 03 Use out of fold margin stacking

Status: Implemented. Decision: Feed group held out margin predictions into classifier and residual scale fitting, refitting learned features inside each fold. Alternative: Use in sample predicted margins. Rationale: In sample fit quality would overstate the value of the auxiliary predictor. Consequence: More training cost and an OOF versus inference feature distribution difference. Revisit if prospective calibration or classifier behaviour deteriorates. Evidence: phm/models.py.

### SD 04 Select with a joint development utility

Status: Implemented. Decision: Maximize U = 0.5 * (AUC - Brier) - 0.5 * CRPS / s, where s is the initial fitting margin standard deviation with a numerical floor of 0.000001. Alternatives: MAE alone, classification accuracy alone or a business cost function. Rationale: Both classification probabilities and margin distributions matter. Consequence: Equal numerical weights are a modelling choice, not an approved operational tradeoff or official combined challenge score. Revisit after stakeholders agree costs and review actions. Evidence: phm/evaluate.py and run.py.

### SD 05 Compete a reduced feature candidate

Status: Implemented. Decision: Retain all raw inputs and up to 22 positively ranked derived attributes using selection set permutation AUC, then compare that reduced candidate with its parent. Training mutual information is descriptive only. Alternatives: No reduction or nested feature selection. Consequence: Correlated features and one reused selection set can make rankings unstable and candidate selection optimistic. The final holdout still provides separate internal assessment. Revisit using repeated development splits or nested validation if selection instability affects results. Evidence: run.py.

### SD 06 Calibrate probability and distribution scale

Status: Implemented. Decision: Use logistic calibration of clipped probability logits and a scalar standardized residual correction on calibration rows. Alternatives: Uncalibrated probabilities, isotonic calibration or conformal intervals. Rationale: Smooth probability calibration and positive scale support the challenge probability output. Consequence: Calibration, scale correction and threshold share one calibration cohort; metrics on that cohort are not independent. Gaussian tails and shifted coverage are unproven. Revisit using an independent operational calibration assessment. Evidence: phm/models.py.

### SD 07 Separate conventional flags from submission actions

Status: Implemented. Decision: Select the conventional threshold by F2 over 181 thresholds from 0.05 to 0.95. Submission class and confidence instead maximize the implemented expected challenge classification score. Alternatives: A fixed probability of 0.5 or one action for both uses. Consequence: class_conf is score optimized and must not be presented as calibrated fault probability. F2 is recall oriented but is not an approved maintenance cost ratio. Revisit conventional actions with operational owners; preserve baseline threshold during shadow evaluation. Evidence: phm/models.py and phm/evaluate.py.

### SD 08 Report multiple performance dimensions

Status: Implemented. Decision: Report discrimination, calibration, confusion counts, margin errors, normal scoring rules and interval coverage together. Alternatives: Accuracy alone or an unsupported combined organizer score. Consequence: High accuracy cannot hide missed faults or narrow unreliable intervals. The organizer's regression normalization is insufficiently specified for an official combined score claim. Revisit only with an authoritative scoring specification. Evidence: phm/evaluate.py and run.py.

### SD 09 Bootstrap duplicate groups

Status: Implemented, limited inference. Decision: Use 200 group bootstrap repeats in the full configuration for selected metrics. Alternative: Row bootstrap or engine bootstrap. Consequence: Intervals account for exact repeats but not hidden engine dependence, model selection variability or uncertain operational sampling. They are not safety guarantees. Revisit with engine or flight level resampling and enough independent assets. Evidence: phm/evaluate.py.

### SD 10 Diagnose operating support loss

Status: Implemented, exploratory. Decision: Remove entire operating regimes from fitting and calibration, then evaluate matching final rows with the chosen configuration. A separate regime investigation compares whole group random removal and fixed thresholds. Consequence: These are shift proxies, not recovered engine identities or confirmatory generalization estimates. Diagnostic refits do not replace the frozen baseline. Revisit with prospective gaps or an engine independent dataset. Evidence: run.py, investigate_regime.py and reference_results/regime5_summary.

### SD 11 Use frozen reference reliability checks

Status: Implemented with provisional policy. Decision: Flag raw range departures, joint and environmental nearest neighbour distances, unusually wide intervals and inadequately covered calibration regimes. Distance and width thresholds use calibration 99th percentiles. Regime coverage checks require at least 100 rows and flag 90 percent coverage below 0.85. Alternatives: No monitoring or a learned error probability monitor. Consequence: These flags indicate reference support concerns; they do not quantify failure probability or guarantee coverage. Revisit with operational alert burden and error association. Evidence: operational/core.py and frozen policy.json.

### SD 12 Evaluate delayed labels against saved predictions

Status: Implemented. Decision: Join exported labels to immutable predictions without rerunning inference. Include eligible labelled valid rows; exclude exact fitting or calibration measurement fingerprints from metrics. Report unlabelled, rejected and seen row counts. Alternatives: Evaluate against a newer model or silently omit unmatched labels. Consequence: Exclusions and selective label availability may bias the eligible cohort. Partial exports are snapshots; cumulative coverage requires cumulative exports. Revisit after defining sampling, label adjudication and completeness targets. Evidence: operational/delayed.py and operational/core.py.

### SD 13 Prioritize review without outcome leakage

Status: Provisional implemented ordering. Decision: P1 is fault flag with reliability warning; P2 is supported fault flag or rejected measurements; P3 is warning without fault flag; P4 has no implemented trigger. Sort within tiers by fault probability then ID. Alternatives: Probability only or residual driven ranking. Consequence: Known outcomes and residuals never drive production queue priority. Tiers are workload ordering, not fault severity. Revisit with engineering owners and prospective workload measurements. Evidence: operational/review.py.

## 7 Results and interpretation

| Internal holdout metric | Uploaded full run | Bundled sample |
|---|---:|---:|
| Evaluated rows | 111381 | 2999 |
| Accuracy | 99.7809 percent | 98.9663 percent |
| Fault recall | 99.8843 percent | 98.6043 percent |
| False negatives | 52 | 17 |
| False positives | 192 | 14 |
| Margin MAE in percentage points | 0.5354 | 0.7976 |
| Margin RMSE in percentage points | 0.7727 | 1.1626 |
| 90 percent interval coverage | 90.6259 percent | 91.6305 percent |

The uploaded full run chose a reduced combined direct model with threshold 0.245. Strong internal scores coexist with sensitivity to missing operating support. In the original regime 5 exclusion experiment, recall fell to about 89.70 percent and 90 percent interval coverage to about 64.35 percent. The normal full model had that regime in training; these results do not justify rejecting every regime 5 observation in normal operation.

The separate reproduction has 24456 evaluation rows rather than the original 24485, so it is explicitly a reproduction rather than exact recovery of the original clustering. On those common rows the main model missed 11 faults, the regime excluded model missed 617, and a size matched random removal control missed 4. Fixed threshold comparisons also show a large gap. Almost all excluded observations remain within retained marginal ranges, while joint nearest neighbour gaps are substantial. This supports missing joint condition coverage as an explanation; unknown engine effects, one random control seed and exploratory slicing limit causal conclusions.

No external challenge accuracy or operational accuracy is available. A replay's reliability warning fraction is not a failure rate. Single class cohorts have undefined ROC AUC and cannot establish both class performance. R squared is unavailable with fewer than two rows. Zero observed positives must be reported with its denominator rather than interpreted as demonstrated fault detection.

## 8 Operational validation design and release criteria

The next confirmatory cohort should contain real engine and flight IDs, observation and label confirmation timestamps, confirmed sensor units, quality indicators, fault evidence and torque margin ground truth. Agree intended review actions and acceptance limits before examining operational outcomes. The acceptance template currently contains unapproved null thresholds; there is no automatic statistical release decision.

Proposed protocol: keep the baseline, feature definitions, calibration, threshold and monitoring policy fixed; make predictions before labels are exposed; retain all observations including rejected and unlabelled ones; record label availability and sampling reasons; evaluate overall, engine, flight, regime and reliability cohorts. Flight summaries currently require complete eligible labels and use any fault versus any flag. They do not implement persistence, cooldown, alert lead time or event onset scoring.

Proposed independent assessment should use engine or flight grouped intervals, report denominators and uncertainty for rare events, compare alert burden with reviewer capacity, and examine warnings against actual errors. Minimum independent assets and fault positive flights must be approved, not chosen from a convenient row count. Uneven label availability needs sensitivity analysis. Timestamp ordering alone does not prove genuinely prospective collection.

Retraining creates a new candidate in a new directory, preserving the old baseline and prediction history. Promotion requires independently evaluated operational performance, agreed acceptance criteria and an explicit release decision. No online update, automatic promotion or automatic threshold tuning is implemented.

## 9 Evidence index and review obligations

Implementation evidence is relative to the complete harness root: phm/data.py; phm/features.py; phm/models.py; phm/evaluate.py; phm/eda.py; run.py; configs/full.yaml; FEATURE_CATALOG.md; investigate_regime.py; operational/core.py; operational/delayed.py; operational/review.py; operational/templates/data_contract.json; operational/templates/acceptance_criteria.json. Result evidence: runs/verified_sample, reference_results/user_full_run and reference_results/regime5_summary. Verification evidence: verify_run.py, verify_pilot.py, verify_investigation.py, tests and validation/tests_latest.log.

The milestone records 34 passing tests in the pinned sample environment, verified labelled and unlabelled replays, and delayed label evaluation on 2999 saved predictions. Tests establish software contracts, not operational suitability. Full artifact execution in its matching 1.9.1 environment, labelled operational validation and review policy approval remain open.

Recommended review responsibilities are modelling design to the modelling owner, data semantics and lineage to the data owner, acceptance thresholds and actions to maintenance engineering, and independence and statistical interpretation to validation. Record actual names, approval dates and evidence cohort IDs when assigned. Revisions should cite the superseded SD identifier, the changed assumption and the new independent evidence. Pair this record with the Architecture Decision Record.

## 10 Quantitative definitions and analysis controls

Let p be calibrated fault probability, y the binary outcome, m the true margin, mu the predicted mean and sigma the calibrated positive scale. The 90 percent interval is mu plus or minus 1.644854 times sigma; the 95 percent interval uses 1.959964. Coverage is the fraction of true margins inside the corresponding interval, and width is the mean upper minus lower bound. Neither empirical calibration nor these Gaussian intervals guarantee conditional coverage.

Brier is the average squared difference between p and y. ECE uses ten equal width probability bins and weights the absolute observed versus predicted probability difference by bin count. Normal negative log likelihood and CRPS are smaller when the predictive distribution better describes outcomes. MAE and RMSE are in margin percentage points. Classification reports a fixed two by two confusion matrix, including cohorts with only one observed class. Undefined ROC AUC and PR AUC are null; the implementation's zero division convention for faulty recall is zero, which must be interpreted with the positive count.

For submission confidence c, a correct class receives c, a false positive receives minus c, and a false negative receives minus c minus 4 times c to the power 11. The action maximizes its expected score conditional on p, rather than copying probability into class_conf. The reported published classification score uses this action; conventional threshold metrics use the separately calibrated F2 threshold. They measure different decisions.

Initial EDA examines distributions, exact repeats, label consistency, predictor relationships and descriptive external shift. Internal environmental slices use initial fitting quartile boundaries and require at least 30 evaluation rows. These slices and the later regime condition investigations are exploratory, with no multiple comparison correction. Small or selected subgroups must not become release claims. Candidate search is a specified comparison rather than a comprehensive hyperparameter optimization; one split seed cannot establish selection stability.

Future statistical acceptance should predefine the operational cohort, exclusions, label sampling and primary metrics, alongside minimum independent engine and flight counts. Report uncertainty and both overall and support flagged cohort performance. Any proposed numerical acceptance threshold must have an owner, intended review action and independent evidence; this record deliberately does not convert the current high internal accuracy into an operational release rule.
