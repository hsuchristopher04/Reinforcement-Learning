# Bundled models and evidence

This directory is outside the ignored `runs/` folder and should be included when distributing the project.

| Model | Original source |
| --- | --- |
| `models/teacher.pt` | `runs/ablation-mse-seed100/best.pt` |
| `models/tree4.json` | `runs/distillation-v1/selected_tree.json` |
| `models/tree8.json` | `runs/capacity-v1/selected_tree.json` |

`manifest.json` records each packaged file's source, byte size, and SHA-256 hash. Run `python project.py verify` from the project folder to check integrity.

`results/` contains copied metrics, configurations, tree rules, reports, and figures. Historical absolute paths in those records describe the author's original runs; evaluation does not depend on those paths. Raw demonstration arrays and intermediate checkpoints are omitted. See the root reproduction guide to regenerate them.
