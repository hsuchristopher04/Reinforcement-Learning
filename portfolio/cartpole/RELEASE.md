# Release checks — October 1, 2026

## Completed

- Installed dependencies in a separate Windows Python 3.13 virtual environment, leaving the original experiment environment intact.
- Replaced the release's PyTorch 2.8.0 pin with 2.14.1 after identifying the upstream checkpoint-loading vulnerability documented in SECURITY.md.
- Upgraded the new environment's pip to 26.2.1. The setup instructions now upgrade pip before installing project dependencies.
- `pip-audit` reported no known vulnerabilities in the completed release environment. This is a point-in-time advisory check, not a guarantee.
- All 18 Python tests passed in the new environment, including tests for artifact path traversal and malformed tree references.
- All four Node playback checks passed: original, aggregation, weighted aggregation, and capacity.
- All 139 packaged artifact hashes matched. Quick-start evaluation now checks integrity before loading models.
- Release-file credential-pattern checks found no matches; local HTML links resolved. The scanner does not detect every possible secret.
- Bandit scanned the Python source with no scan errors and no medium/high findings. Its 27 low-severity findings comprise 25 research/trajectory assertions and two subprocess warnings. Assertions check experiment invariants and are not authorization controls; run research scripts without Python's `-O` option. The subprocess calls invoke fixed local scripts through the current Python interpreter with argument lists, without a shell.

## Patched-environment evaluation

100 episodes per policy, reset seeds starting at 18000000:

| Policy | Mean return | Completion |
| --- | ---: | ---: |
| Teacher | 500.00 | 100% |
| Four-leaf tree | 480.38 | 67% |
| Eight-leaf tree | 497.89 | 97% |

This is a release smoke evaluation on a subset of the original capacity-test seeds, not a new independent research result or a replacement for the 500-episode study.

## Scope and remaining checks

The review covers the portfolio project, not a security audit of the existing coursework repository, Python itself, or every dependency's source. Historical experiments were not retrained. Original evidence and checkpoints are unchanged. The old `.venv` still contains historical dependencies; upgrade it with the README commands before using it for new work, or use the isolated release environment under `runs/release-venv`.

Automated browser checks use a mock DOM. Final visual review in a real browser remains outstanding because browser access to the local demo was blocked. Inspect the landing page and all four demos at desktop and narrow widths before announcing the release.

Publishing was attempted but blocked by authentication: Git has no saved credentials, and the connected GitHub integration returned HTTP 403, `Resource not accessible by integration`, on a blob upload. No remote release changes were made. The prepared checkout is under `runs/release-repository`; project files go in `portfolio/cartpole/`, preserving all coursework folders. Sign in to GitHub through Git Credential Manager or GitHub Desktop to publish the prepared commit. Never paste an access token into chat or source files.

Raw audit reports and the installed package inventory are retained locally under `runs/release-audit.json`, `runs/release-bandit.json`, and `runs/release-environment.txt`.
