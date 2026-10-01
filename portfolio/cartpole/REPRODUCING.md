# Reproducing the project

## Evaluate saved policies

Follow the README environment setup, then run:

```powershell
python project.py verify
python project.py evaluate all --episodes 100
```

This path needs no `runs` directory. Five episodes provide a fast smoke check; 100 or 500 provide a larger evaluation. Use `--seed` to specify the first reset seed. Historical reports use their recorded evaluation seeds, so a fresh evaluation need not match their numbers.

## Rebuild the research pipeline

Use a fresh project copy without existing experiment outputs for the canonical commands below. Keep your original experiments. Scripts refuse to overwrite existing output directories; do not delete valuable results to rerun them. The full pipeline is substantially longer than evaluating saved models.

With the environment activated and the project as your working directory:

```powershell
python cartpole.py train --seed 100 --steps 50000 --loss mse --output runs/ablation-mse-seed100
python distill.py run --teacher runs/ablation-mse-seed100/best.pt --output runs/distillation-v1
python dagger.py
python weighted_dagger.py
python capacity.py
python replication.py
```

The later scripts consume the preceding canonical run folders. Inspect each study's documentation for configuration and seed splits. This sequence covers the core research pipeline, not every exploratory teacher ablation. Training results can vary across environments.

To study distillation while holding the teacher fixed, replace the first command by copying `artifacts/models/teacher.pt` to `runs/ablation-mse-seed100/best.pt` in that fresh copy, creating its parent directory first. Raw demonstration arrays and intermediate checkpoints are not bundled: subsequent stages regenerate them in order.

The convenience commands `python project.py train` and `python project.py distill` create timestamped runs. Those are useful for new experiments but do not automatically populate the canonical paths above.

## Rebuild recorded demonstrations

After producing the required experiment outputs:

```powershell
python record_demo.py
python record_aggregation_demo.py
python record_aggregation_demo.py --weighted
python record_capacity_demo.py
```

These commands regenerate the existing demo folders. `package_artifacts.py` is a maintainer command that replaces the curated artifact snapshot from canonical runs; use it only when intentionally updating the distributed evidence and models. Normal evaluation does not require it.

## Checks and environment

```powershell
python project.py verify
python -m unittest discover -p "test_*.py" -v
node test_demo.cjs
node test_aggregation_demo.cjs
node test_aggregation_demo.cjs --weighted
node test_aggregation_demo.cjs --capacity
```

The recorded dependency environment is Windows, Python 3.13, and CPU evaluation. Install from `requirements.txt`; `requirements-tested.txt` preserves historical experiment versions, including an older vulnerable PyTorch, and should not be used for normal installation. See SECURITY.md and RELEASE.md for release validation. GPU and rendering dependencies are unnecessary for saved-policy evaluation and the offline demos.

The JavaScript checks exercise demo logic with a mock DOM. Actual visual review and a fresh dependency installation are separate release checks.
