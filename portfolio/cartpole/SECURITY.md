# Security and release scope

This is a local research program and a static demo, not a service that accepts public uploads. The demo replays bundled data and needs no accounts, API keys, external scripts, or backend. Do not expose a development server containing your entire working directory.

## Model and dataset trust

Only load checkpoints and datasets from sources you trust. PyTorch checkpoints are not a safe interchange format for arbitrary uploads. This project uses explicit `weights_only=True`, but older PyTorch releases had vulnerabilities even in that mode. Release requirements replace the original 2.8.0 pin with 2.14.1. See [the upstream checkpoint advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).

The quick-start evaluator verifies the bundled artifact hashes and sizes before loading policies. This detects changed files relative to the local manifest; it does not authenticate a manifest that an attacker has also replaced. Obtain the code, models, and manifest together from the trusted repository.

Research dataset loaders disable NumPy pickle loading. Tree traversal rejects cyclic or out-of-range child references. These checks do not make arbitrary large or hostile files safe to load: memory exhaustion and malformed inputs remain possible. Research scripts are intended for trusted local inputs.

## Website and commands

Generated JSON embedded in HTML escapes `<` to prevent data from closing its script element. Playback renders labels with text nodes. CLI forwarding uses an argument list without a shell. The public demo contains recorded trajectories and aggregate results, not credentials or private user submissions.

Run `python release_check.py` for local-link and credential-pattern checks, `python project.py verify` for bundled integrity, and the tests in the reproduction guide. Pattern-based scans and dependency advisories cannot prove the absence of vulnerabilities. Repeat dependency auditing before future releases.

Historical result configurations preserve the original environment and local file paths as provenance. `requirements-tested.txt` records that historical environment, including the old PyTorch version: do not install it for normal use. Use `requirements.txt` for the release dependencies.

## Reporting

Report a suspected vulnerability privately to the repository owner through an available private contact or GitHub private vulnerability reporting if enabled. Do not include secrets or exploit payloads in a public issue.
