# PHM 2024 engine-health harness

A local Python workspace for the [PHM Society 2024 Data Challenge](https://phmsociety.org/): describe the data, train comparable models, freeze the one you trust, then review its predictions before any label arrives.

Everything you run lives in [`phm2024_cursor_harness`](phm2024_cursor_harness). Open that folder. Commands below assume you are inside it.

## What it answers

Each row is one observation of an engine with seven measurements (`trq_measured`, `oat`, `mgt`, `pa`, `ias`, `np`, `ng`). The harness learns two things together:

1. **Is this observation faulty?** A calibrated probability, plus a class label chosen for the challenge’s asymmetric score.
2. **How much torque margin is left?** A predicted value and a positive uncertainty, exported as a normal distribution.

A local dashboard then puts those predictions in a review queue. Review notes never overwrite the model, the probabilities, or the labels.

## Pick a path

| You want to… | Start here |
|---|---|
| See the included sample model and the review dashboard | [Path A below](#path-a--look-at-the-sample-in-about-five-minutes) |
| Train a new experiment from the challenge CSVs | [Path B below](#path-b--train-a-new-experiment) |
| Keep an existing full model frozen and score labels later | [Path C below](#path-c--predict-now-labels-later) |
| Change the code with Cursor | [`CURSOR_TASK.md`](phm2024_cursor_harness/CURSOR_TASK.md) and [`AGENTS.md`](phm2024_cursor_harness/AGENTS.md) |

The longer map of every command is [`START_HERE.md`](phm2024_cursor_harness/START_HERE.md). Feature definitions and the assumptions behind them are in [`FEATURE_CATALOG.md`](phm2024_cursor_harness/FEATURE_CATALOG.md).

## Path A — look at the sample in about five minutes

The bundled sample was trained on 20,000 rows and saved with **Python 3.12** and **scikit-learn 1.8.0**. Use that environment for the demo. A model trained in another environment will refuse to load, on purpose.

```powershell
cd phm2024_cursor_harness
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\Activate.ps1
python -m pip install -r requirements-pilot-demo.txt
python -m unittest discover -s tests -v
python pilot.py doctor --baseline pilot_artifacts/sample_demo
python pilot.py serve --baseline pilot_artifacts/sample_demo --run-dir runs/pilot_replay_demo --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) and leave the terminal running. Ctrl+C stops the server.

On macOS or Linux, create the environment with `python3.12 -m venv .venv-demo` and activate it with `source .venv-demo/bin/activate`.

The sample is a demonstration of the workflow. It is not a full-data result. The labelled replay uses challenge holdout rows; it is not operational engine history.

## Path B — train a new experiment

Place the four challenge files in `phm2024_cursor_harness/data/`:

- `X_train.csv`
- `y_train.csv`
- `X_test.csv`
- `X_validation.csv`

Those CSVs are not stored in git. Copy them in locally.

```powershell
cd phm2024_cursor_harness
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\Activate.ps1
python -m pip install -r requirements-pilot-demo.txt

# Summaries only. No model is fit.
python run.py --stage eda --data-dir data --output-dir runs/eda

# Same recipe, 60,000 training rows. Good first experiment.
python run.py --config configs/default.yaml --data-dir data --output-dir runs/quick

# All labelled training rows. This is the expensive run.
python run.py --config configs/full.yaml --data-dir data --output-dir runs/full

python verify_run.py --run-dir runs/full --data-dir data
```

Every `--output-dir` must be a new empty folder. Read `runs/<name>/REPORT.md` first. `selected_model.joblib` in that folder is the exact model that was evaluated.

`configs/default.yaml` and `configs/full.yaml` compare the same candidates: a linear margin baseline, then histogram gradient boosting on raw, physics-informed, statistical, and combined features. Physics features that treat pressure altitude as altitude are off unless you opt in with `configs/altitude_m_hypothesis.yaml`. Official descriptions and published papers disagree on what `pa` means, so the default leaves it opaque.

## Path C — predict now, labels later

Freeze a model you already trust, score a new CSV, and only later attach confirmed labels. Evaluation reads the saved predictions. It does not reload the model or retrain.

```powershell
python pilot.py freeze --model runs/full/selected_model.joblib --training-csv data/X_train.csv --output pilot_artifacts/full_baseline --name full-selected-frozen
python pilot.py shadow --baseline pilot_artifacts/full_baseline --input operational_data/observations.csv --contract operational_data/data_contract.json --output runs/shadow_predictions_001
python pilot.py serve --baseline pilot_artifacts/full_baseline --run-dir runs/shadow_predictions_001 --port 8765
```

When a label export exists:

```powershell
python pilot.py evaluate --baseline pilot_artifacts/full_baseline --run-dir runs/shadow_predictions_001 --labels operational_data/labels.csv --output runs/shadow_evaluation_001
```

Use the Python environment that trained that model. The sample demo pins scikit-learn 1.8.0; a full run may have been saved with a different version. `pilot.py doctor` checks this before it unpickles anything.

Review priority is a queue order, not a safety rating:

| Tier | When it appears |
|---|---|
| P1 | Fault flag and a reliability warning |
| P2 | Fault flag alone, or a rejected input |
| P3 | Reliability warning without a fault flag |
| P4 | Nothing in the current rules fired |

## How a training run is split

Duplicate feature rows stay in the same partition. Targets and row ids are never used as inputs.

| Partition | About | Used for |
|---|---:|---|
| Train | 60% | Fit features and candidates |
| Selection | 15% | Choose the candidate |
| Calibration | 10% | Calibrate probabilities and the margin scale |
| Final holdout | 15% | Score the frozen model once |

The holdout is not folded back into the saved model. That keeps the reported numbers tied to one artifact.

## What is in this repository

```
phm2024_complete_harness/
├── README.md                      ← you are here
└── phm2024_cursor_harness/
    ├── run.py                     train, compare, export a submission
    ├── pilot.py                   freeze, predict, review, score late labels
    ├── verify_run.py              check a training run
    ├── verify_pilot.py            check a frozen prediction run
    ├── configs/                   quick, full, and altitude-hypothesis settings
    ├── phm/                       data, features, models, metrics
    ├── operational/               dashboard, review queue, delayed labels
    ├── tests/                     contract and workflow tests
    ├── pilot_artifacts/           frozen baselines (model files stay local)
    ├── reference_results/         metrics from a completed full run
    └── data/                      your challenge CSVs (not in git)
```

## What stays on your machine

Git ignores these so a clone stays source code plus small reference outputs:

| Left out | Why |
|---|---|
| `.venv-pilot/`, `.venv/`, `.venv-demo/` | Local environments. Recreate them from the requirements files. |
| `data/*.csv` | Challenge tables. Copy the four CSVs into `data/` yourself. |
| `runs/` | Experiment outputs. Each run should be produced locally. |
| `*.joblib` | Saved models and reliability sidecars. Load them only from a source you trust. |

`reference_results/` is included. It records metrics from a completed full run. It is not a model you can serve.

## Requirements

- Python 3.12 for the pinned sample (`requirements-pilot-demo.txt`)
- Python 3.11+ for a fresh training environment (`requirements.txt`)
- Packages: NumPy, pandas, SciPy, scikit-learn, Matplotlib, joblib, PyYAML

No network service and no API key are required. The dashboard binds to localhost.

## Limits worth knowing before you cite a number

- Internal scores are an explicit utility used to pick a candidate. They are not an official challenge leaderboard score.
- Engine identifiers were removed, so a good holdout score is not proof the model works on an unseen engine.
- Reliability warnings and queue tiers are provisional. They are not operating limits or maintenance instructions.
- A reviewer’s note is not a diagnosis. Confirmed labels have to come from a separate export.
- Negative torque margin is not, by itself, the fault definition. Healthy and faulty labels both occur on either side of zero.
