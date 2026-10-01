# Weighted aggregation experiment

Both controls reproduced their reference trees exactly.

| Weight | Validation mean | Completion | Position failures | Angle failures |
|---|---:|---:|---:|---:|
| 0.0 | 483.40 | 66.0% | 68 | 0 |
| 0.1 | 483.55 | 66.0% | 68 | 0 |
| 0.25 | 11.64 | 0.0% | 0 | 200 |
| 0.5 | 11.60 | 0.0% | 0 | 200 |
| 1.0 | 11.59 | 0.0% | 0 | 200 |

Continued aggregation: False. Selected weight 0.1, round 1.

| Policy | Test mean | Completion | Minimum |
|---|---:|---:|---:|
| baseline | 485.20 | 66.8% | 363 |
| weighted | 485.12 | 66.8% | 363 |
| teacher | 500.00 | 100.0% | 500 |

![Weight comparison](weight_comparison.png)

500 matched fresh test seeds; selection completed before test evaluation. One fitting seed and dataset. Original validation examples were excluded from fitting. No post-test tuning occurred.
