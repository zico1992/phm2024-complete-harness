# Operational data intake request

Collection status: awaiting source access and data-owner confirmation. No external messages
have been sent. The implementation can validate and evaluate supplied operational CSVs now.

| Owner to involve | Required data/decision | Why it is needed |
|---|---|---|
| Engine/OEM performance engineer and challenge data owner | Definitions and units of torque, OAT, MGT, PA, IAS, NP, NG; design torque reference and margin definition | Align operational measurements with the frozen baseline and resolve PA/NP ambiguity |
| Telemetry/data engineering | Stable observation/engine/flight IDs, actual UTC timestamps, sampling/aggregation windows, sensor-quality flags, calibration history | Trace predictions, reject bad data and establish real engine/flight chronology |
| Maintenance/reliability engineering | Confirmed fault diagnosis, inspection findings, label source and confirmation date, event/onset times where known | Independent labels; fault sign is not inferred from margin sign |
| Fleet operations and maintenance planning | Intended review action, acceptable missed faults/false alerts, review capacity, pilot asset scope | Approve useful operating criteria and review workload |
| Data owner/IT | Authorised export location, access and retention requirements | Obtain and retain the labelled cohort with provenance |

Request a consecutive cohort covering normal and confirmed faulty operation across multiple
engines, flights and OAT/PA/IAS conditions. Do not export only triggered alerts: this prevents
estimating missed faults or overall calibration. Include every eligible observation in each
flight and document exclusion criteria before evaluation.

Minimum delivered files: operational/templates/observations.csv and labels.csv populated,
confirmed data_contract.json and agreed acceptance_criteria.json. Confirm fault labels describe
the intended maintenance condition rather than a different fault taxonomy. Record known
unlabelled/uncertain cases explicitly; do not substitute engineer review dispositions for truth.
Request identifiers linking the historical challenge engines to operational engines if available.
Without them we cannot establish that an operational engine is unseen by the baseline.

Pilot process: validate intake, freeze cohort/hashes, score in shadow, review errors and support,
measure engine/flight workload, record inspection outcomes, and decide against pre-agreed
criteria. If measurements have incompatible semantics or unsupported conditions, resolve the
contract or start a new development model experiment while retaining the frozen baseline.
