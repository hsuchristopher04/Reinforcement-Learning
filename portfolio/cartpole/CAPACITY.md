# Eight leaves improve reliability

The smallest candidate meeting the pre-specified 90% validation completion
target was an **eight-leaf tree trained on the original demonstrations**.
On 500 fresh matched test episodes it completed **98.4%**, compared with **70.8%**
for the saved four-leaf baseline. The baseline and all earlier demos remain intact.

## Experiment

Ten candidates compared leaf budgets 4, 8, 16, 32, and 64 across two fixed datasets:
80,000 original teacher examples, or those examples plus 24,295 first-round
learner-state examples with equal weights. The original 20,000 action-validation
examples were excluded. Fitting seed was 2026 throughout.

`max_leaf_nodes` controls size directly and uses best-first growth. It is not
equivalent to the earlier depth cap: the new four-leaf original-data tree has
depth 3, while the saved four-leaf baseline has depth 2. The saved baseline was
therefore evaluated separately, not treated as the new four-leaf candidate.

## Validation: 200 shared reset seeds

| Dataset | Leaves | Depth | Mean return | Completion |
|---|---:|---:|---:|---:|
| Original | 4 | 3 | 201.73 | 0% |
| **Original** | **8** | **4** | **498.14** | **98.5%** |
| Original | 16 | 8 | 499.55 | 99.5% |
| Original | 32 | 8 | 295.56 | 31% |
| Original | 64 | 12 | 500.00 | 100% |
| Augmented | 4 | 2 | 11.39 | 0% |
| Augmented | 8 | 5 | 9.36 | 0% |
| Augmented | 16 | 7 | 375.31 | 57.5% |
| Augmented | 32 | 10 | 135.43 | 3.5% |
| Augmented | 64 | 13 | 291.64 | 37.5% |

The saved baseline completed 69% on these validation seeds. Selection chose
the smallest actual leaf count meeting 90%, with completion and then mean return
breaking ties. This selected original-data budget 8 before any test evaluation.

![Completion versus leaves](artifacts/results/capacity-v1/capacity_comparison.png)

## Final test: 500 untouched matched episodes

| Policy | Mean return | Episode std | Minimum | Completion |
|---|---:|---:|---:|---:|
| Saved four-leaf baseline | 485.66 | 29.23 | 336 | 70.8% |
| Selected eight-leaf tree | 498.84 | 11.87 | 288 | 98.4% |
| Teacher | 500.00 | 0.00 | 500 | 100% |

The eight-leaf tree failed in 8 episodes, so it is not perfectly reliable. Its
lowest return was also lower than the baseline's minimum. The gain is in
completion frequency and mean return, not every possible outcome.

More capacity did not improve performance monotonically: the 32-leaf
original-data candidate was worse than both the 16- and 64-leaf candidates.
Augmentation performed worse than original data at every tested leaf budget.
The evidence supports the selected eight-leaf policy for this setup, not a claim
that four leaves cannot succeed or that larger trees always work better.

This experiment uses one dataset pair and one fitting seed. Reset seeds starting
at 17,000,000 were used for validation; seeds 18,000,000–18,000,499 for testing.
No additional fitting or selection followed the test results.

## Demo and reproduction

Open **demo-capacity/index.html**. The Original / Selected selector compares
four and eight leaves. The examples include a baseline failure the larger tree
fixes, a shared success, and a remaining larger-tree failure. Each variant uses
the same example seeds, with the teacher alongside it. The eight-leaf tree is
shown in full in a responsive diagram with exact recorded path highlighting.
Other tree displays retain their existing layout.

```powershell
# Evaluate the selected model on another set of reset seeds.
.\.venv\Scripts\python.exe distill.py evaluate runs/capacity-v1/selected_tree.json --episodes 100 --seed 19000000

# Reproduce the experiment without overwriting the existing run.
.\.venv\Scripts\python.exe capacity.py --output runs/capacity-reproduction

# Regenerate the separate demo.
.\.venv\Scripts\python.exe record_capacity_demo.py

# Selection and playback logic checks.
.\.venv\Scripts\python.exe test_capacity.py -v
node test_aggregation_demo.cjs --capacity
```

Full metrics, failure counts, all ten models/rule files, input hashes, source
snapshots, selection metadata, and test returns are in `runs/capacity-v1/`.
Trajectory checks and mock-DOM playback checks cover logic; browser layout is
not automatically verified.
