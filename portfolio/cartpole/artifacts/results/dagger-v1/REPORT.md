# Dataset aggregation at a four-leaf budget

## Failure diagnosis

{"episodes": 100, "failures": 35, "position_failures": 35, "angle_failures": 0, "all_state_disagreement": 0.41320359343554847, "failure_tail_disagreement": 0.36857142857142855}

## Validation rounds

| Round | Added examples | Total examples | Leaves | Mean | Minimum | Completion |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 24295 | 104295 | 4 | 11.25 | 9 | 0.0% |
| 2 | 555 | 104850 | 4 | 11.28 | 9 | 0.0% |
| 3 | 586 | 105436 | 4 | 11.28 | 9 | 0.0% |
| 4 | 577 | 106013 | 4 | 11.25 | 9 | 0.0% |
| 5 | 563 | 106576 | 4 | 11.25 | 9 | 0.0% |

Selected round 2 using validation only; baseline validation completion 71.0%.

## Fresh test: 500 matched seeds per policy

| Policy | Mean | Std | Minimum | Completion |
|---|---:|---:|---:|---:|
| baseline | 485.49 | 27.27 | 357 | 68.0% |
| aggregated | 11.75 | 3.37 | 9 | 0.0% |
| teacher | 500.00 | 0.00 | 500 | 100.0% |

![Validation learning curve](learning_curve.png)

## Scope

One aggregation run and fitting seed; episode variation does not measure training-seed variability.
Teacher labels were queried on learner states; learner actions were executed. Diagnostic, collection, validation, and test seed ranges are disjoint.
Only the original 80,000 training examples were aggregated. The 20,000 original validation examples were excluded.
All examples have equal weight. Later rounds accumulate the errors and distributions of earlier learners.
Five rounds were pre-specified. No additional tuning or tree-size increase followed test evaluation.
Disagreement near failures is descriptive, not proof that a particular action caused failure.
