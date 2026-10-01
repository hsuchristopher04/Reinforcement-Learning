# Explainable CartPole

How much decision-tree complexity is needed to imitate a strong reinforcement-learning policy reliably? This project trains a neural CartPole teacher, distills it into small trees, and studies both successful policies and failure cases.

Read the [portfolio case study](PORTFOLIO.md) for the experimental story, key results, limitations, and a two-minute demonstration walkthrough.

Open [index.html](index.html) in a browser to explore the saved demonstrations. They work offline without Python. GitHub's source viewer does not run HTML; download the project to open the page locally.

## Quick start

From the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe project.py verify
.\.venv\Scripts\python.exe project.py evaluate all --episodes 100
Invoke-Item .\index.html
```

Skip environment creation and installation if your environment is already set up. Evaluation uses the three bundled policies in `artifacts/models`; it does not need the ignored `runs` folder. Use `--episodes 5` for a quick smoke check. Evaluation defaults to seed 22000000, so results can differ from the historical studies.

With your environment activated, the main commands are:

| Command | Purpose |
| --- | --- |
| `python project.py verify` | Check bundled model and evidence hashes |
| `python project.py evaluate teacher` | Evaluate the neural teacher |
| `python project.py evaluate tree4` | Evaluate the original four-leaf tree |
| `python project.py evaluate tree8` | Evaluate the selected eight-leaf tree |
| `python project.py train --seed 101` | Train a new teacher in a fresh timestamped run |
| `python project.py distill` | Distill the bundled teacher in a fresh run |
| `python project.py demo` | Print the demo landing-page path |

The training and distillation commands forward options to the original scripts. The distillation command runs the original depth sweep; the separate capacity study produces the selected eight-leaf policy. See [full reproduction instructions](REPRODUCING.md) for that pipeline.

## Findings

On the capacity study's shared 500-episode evaluation, the eight-leaf tree completed 98.4% of episodes, compared with 70.8% for the original four-leaf tree and 100% for the teacher. Their mean returns were 498.84, 485.66, and 500 respectively.

The replication study collected five independent demonstration datasets from the same fixed teacher. Eight-leaf completion rates ranged from 90.8% to 99.6%, averaging 97.12%. Three fitting seeds produced identical policies within each dataset: these are five distinct policies, not fifteen independent replications. These results concern the default simulator, not robustness to changed physics.

![Completion across five demonstration datasets](artifacts/results/replication-v1/replication.png)

Read the experiments in order: [teacher](EXPERIMENTS.md), [distillation](DISTILLATION.md), [aggregation](DAGGER.md), [weighted aggregation](WEIGHTED_DAGGER.md), [capacity](CAPACITY.md), and [replication](REPLICATION.md). Aggregation failures are included because they explain why increasing tree capacity was investigated.

## Project layout

- `index.html` and `demo*/`: offline landing page and recorded interactive demonstrations.
- `artifacts/models/`: neural teacher, four-leaf tree, and eight-leaf tree.
- `artifacts/results/`: curated metrics, configuration, reports, and figures.
- `artifacts/manifest.json`: source paths, sizes, and SHA-256 hashes; see [artifact provenance](artifacts/README.md).
- `project.py`: quick-start entry point; the original research scripts remain available at the root.
- `runs/`: ignored working experiment outputs. Raw demonstration datasets and intermediate checkpoints are regenerated through the reproduction pipeline.
- `test_*.py` and `test_*.cjs`: Python checks and browser-logic checks.

Legacy Pong coursework files remain separate from the CartPole workflow. Saved teacher checkpoints support inference; they are not full training-resume snapshots.

## Validation and dependencies

```powershell
python -m unittest discover -p "test_*.py" -v
```

`requirements.txt` contains the release dependencies, including patched PyTorch. `requirements-tested.txt` preserves the original experiment environment and is historical provenance, not an installation recommendation. See [security notes](SECURITY.md) and [release checks](RELEASE.md). Node checks and full reproduction details are in [REPRODUCING.md](REPRODUCING.md). Mock browser checks do not replace visual review.
