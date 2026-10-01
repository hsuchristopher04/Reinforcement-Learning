# How simple can the CartPole policy be?

A decision tree of depth 2 with four leaves retained a mean return of **483.41
out of 500** on 100 fresh CartPole episodes, compared with the teacher's 500.
This is 96.7% of the teacher's mean return, but not the same reliability: the tree
completed 67% of episodes without failure, compared with the teacher's 100%.

## Method

- Froze `runs/ablation-mse-seed100/best.pt`; no teacher training occurred.
- Collected 200 episodes (100,000 observations and greedy action labels).
- Split entire episodes: 160 training (80,000 examples), 40 for action agreement
  (20,000 examples), with fixed split seed 2026.
- Fit scikit-learn decision trees of maximum depths 1, 2, 3, 4, 5, 6, 8, and 10.
- Evaluated actual tree control on 100 identical validation reset seeds, separate
  from the teacher demonstrations.
- Selected the fewest-leaf tree with validation mean >=475, before inspecting tests.
- Evaluated only the selected tree and teacher on 100 fresh, shared test seeds.

Collection reset seeds start at 6,000,000; control validation at 7,000,000;
the final test at 8,000,000. No reward shaping or additional demonstrations from
tree-visited states were used. This is a behavioral-cloning baseline.

## Validation results

| Maximum depth | Leaves | Teacher-action agreement | Mean return | Full completion |
|---|---:|---:|---:|---:|
| 1 | 2 | 65.50% | 29.28 | 0% |
| **2** | **4** | **76.71%** | **486.84** | **68%** |
| 3 | 8 | 77.73% | 191.48 | 0% |
| 4 | 16 | 82.58% | 174.98 | 0% |
| 5 | 32 | 84.02% | 239.67 | 0% |
| 6 | 63 | 85.78% | 194.13 | 0% |
| 8 | 215 | 90.92% | 463.40 | 73% |
| 10 | 542 | 92.75% | 485.46 | 93% |

Higher action agreement did not consistently yield higher control returns.
The selected tree was chosen for small size subject to the mean-return threshold,
not for the highest completion rate. In particular, depth 10 completed more
validation episodes than depth 2, while their mean returns were similar.

![Validation returns versus tree size](artifacts/results/distillation-v1/complexity_vs_return.png)

## Final test

| Policy | Mean return | Episode standard deviation | Full completion |
|---|---:|---:|---:|
| Frozen teacher | 500.00 | 0.00 | 100% |
| Selected depth-2 tree | 483.41 | 31.27 | 67% |

These results cover one demonstration collection and one fitting seed. They do
not measure variability across datasets or robustness to changed physics.

## What the tree learned

The selected policy uses pole angle and pole angular velocity, ignoring cart
position and cart velocity. Its left subtree contains a redundant split: both
leaves select left. Approximately, its action rules can be expressed as:

1. If pole angular velocity <= -0.334911 rad/s, push left.
2. Otherwise, if pole angle <= -0.020546 radians, push left.
3. Otherwise, push right.

The stored model retains all four original leaves and full-precision thresholds.
These rounded rules explain it; use the JSON model for exact reproduction.

![Selected decision tree](artifacts/results/distillation-v1/selected_tree_overview.png)

## Artifacts and reproduction

The self-contained local run is `runs/distillation-v1/`. It includes the dataset,
episode split, all eight JSON trees and readable rule files, per-episode returns,
selection metadata, final tests, source snapshots, package versions, and teacher
checksum. `runs/distillation-baseline/` contains the initial data collection;
an export bug interrupted that first run, and the completed run reused those
same demonstrations after fixing serialization and threshold comparisons.

```powershell
# Reevaluate the selected policy on another set of initial conditions.
.\.venv\Scripts\python.exe distill.py evaluate runs/distillation-v1/selected_tree.json --episodes 100 --seed 9000000

# Reproduce the entire experiment in a new timestamped directory.
.\.venv\Scripts\python.exe distill.py run

# Check episode separation, saved-tree prediction parity, and selection logic.
.\.venv\Scripts\python.exe test_distill.py -v
```

Run artifacts are Git-ignored; distribute the models, figures, and dataset
separately when publishing the portfolio. The saved trees use JSON rather than
pickled Python objects.

## Next experiment

Improve reliability by collecting teacher labels on states visited by the tree
and retraining (dataset aggregation). Keep this baseline and reserve new test
seeds for that experiment. Alternatively, first build a visual demo showing the
teacher and tree side by side, with the active rule highlighted.
