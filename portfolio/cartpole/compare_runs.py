"""Compare validation curves without reading held-out test results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compare(runs, output, max_steps):
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    summary = []
    for run in runs:
        with (run / "evaluations.csv").open(encoding="utf-8") as source:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(source)]
        rows = [row for row in rows if row["step"] <= max_steps]
        if not rows:
            continue
        best = max(rows, key=lambda row: row["mean_return"])
        summary.append({"run": run.name, "best_validation": best["mean_return"],
                        "best_step": int(best["step"]), "final_validation": rows[-1]["mean_return"],
                        "final_step": int(rows[-1]["step"])})
        ax.plot([r["step"] for r in rows], [r["mean_return"] for r in rows], label=run.name)
    ax.axhline(475, linestyle="--", color="gray", label="Project target")
    ax.set(xlabel="Training environment steps", ylabel="Mean validation return",
           title="CartPole training comparisons (validation only)", ylim=(0, 510))
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=50_000)
    args = parser.parse_args()
    compare(args.runs, args.output, args.max_steps)
