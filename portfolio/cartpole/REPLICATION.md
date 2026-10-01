# Eight-leaf reliability across five demonstration datasets

The fixed eight-leaf training procedure produced completion rates from **90.8%
to 99.6%** across five new demonstration datasets, averaging **97.12%** across
datasets. All five exceeded 90% on the shared 500-episode evaluation set.
This supports reproducibility within the tested setup, while showing sensitivity
to which demonstrations are collected. It does not guarantee 98.4% for every fit.

## All datasets and fitting seeds

| Demonstration dataset | Seed 2026 completion | Seed 2027 completion | Seed 2028 completion | Mean return | Distinct trees |
|---|---:|---:|---:|---:|---:|
| 1 | 90.8% | 90.8% | 90.8% | 492.69 | 1 |
| 2 | 99.0% | 99.0% | 99.0% | 498.77 | 1 |
| 3 | 98.4% | 98.4% | 98.4% | 497.79 | 1 |
| 4 | 97.8% | 97.8% | 97.8% | 498.57 | 1 |
| 5 | 99.6% | 99.6% | 99.6% | 499.25 | 1 |

All three fitting seeds produced an exactly identical serialized policy within
each dataset. There were **five distinct trees across 15 fits**, not 15 independent
policy replications. Exact duplicates reused the deterministic evaluation results
on the same reset seeds; that reuse is recorded in each result file.

![All replication results](artifacts/results/replication-v1/replication.png)

## Frozen protocol

- Same successful neural teacher, unchanged weights and greedy actions.
- Five collections of 200 episodes: 100,000 examples each, 500,000 in total.
- Split each collection by whole episode into 160 training and 40 action-validation
  episodes (80,000 / 20,000 examples). Split seed 2026 is fixed.
- Fit three trees per collection using `max_leaf_nodes=8`, no depth cap, default
  Gini criterion, and fitting seeds 2026, 2027, and 2028.
- Evaluate on the same 500 fresh reset seeds for every distinct policy, the saved
  eight-leaf reference, and the teacher. Report every fitted policy.
- No candidate selection, no new hyperparameter tuning, and no model replacement.

Collection seeds start at 20,000,000, 20,001,000, 20,002,000, 20,003,000,
and 20,004,000. Evaluation uses 21,000,000–21,000,499. All ranges are disjoint
and separate from prior experiments. Original models and demos remain unchanged.

## Interpretation

On the same new evaluation seeds, the saved eight-leaf reference achieved
98.6% completion and 499.30 mean return; the teacher achieved 100% and 500.00.
The five distinct replication policies had 46, 5, 8, 11, and 2 track-boundary
failures respectively, with no pole-angle failures. Remaining reliability
variation was therefore associated with cart drift in these evaluations.

The demonstration dataset affected the learned policy more than fitting-seed
tie-breaking in these runs. Dataset 1 produced a noticeably less reliable policy
than datasets 2–5, despite using the same sample count and fitting procedure.
The five independent collections are the main replication units. The reported
97.12% equally weights their completion rates; it is not a claim of 15 independent
confirmations or thousands of independent evaluation starting conditions.

Evaluation episodes are paired across policies. These results do not test new
physics, wider initial-state distributions, a different teacher, or robustness
to measurement noise. Five datasets give useful evidence, not a universal
guarantee. The most accurate portfolio claim is: **the fixed procedure exceeded
90% completion for all five tested demonstration collections, with measurable
variation between collections.**

## Artifacts and reproduction

`runs/replication-v1/` contains all five datasets and episode splits, all 15
tree/rule files, per-episode evaluation results, action agreement and failure
counts, exact-policy fingerprints, source snapshots, package versions, input
checksums, dataset-level summaries, and the generated full report. Reference
policy results are in `references.json` and `REPORT.md`.

```powershell
# Run the same protocol into a new directory (never overwrite existing results).
.\.venv\Scripts\python.exe replication.py --output runs/replication-reproduction

# Check dataset-level averaging, fingerprinting, and non-overlapping seed ranges.
.\.venv\Scripts\python.exe test_replication.py -v
```

The current eight-leaf demo stays an illustration of the previously selected
policy. No replication winner was chosen to replace it.
