"""Five rounds of pure learner-rollout dataset aggregation at fixed depth two."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.tree import DecisionTreeClassifier, export_text

import cartpole as c
import distill as d

ROOT = Path(__file__).resolve().parent


def collect_labeled(teacher, model, episodes, seed):
    """Execute learner actions; teacher labels exactly those visited states."""
    env = c.gym.make(c.ENV_NAME)
    states, labels, actions, seeds, steps, outcomes = [], [], [], [], [], []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            start = len(states)
            for step in range(500):
                label = c.select_action(teacher, obs)
                action = d.tree_action(model, obs)
                states.append(obs.copy()); labels.append(label); actions.append(action)
                seeds.append(seed + episode); steps.append(step)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    outcomes.append({"seed": seed + episode, "length": step + 1,
                        "completed": bool(truncated and not terminated),
                        "position_failure": bool(abs(obs[0]) > env.unwrapped.x_threshold),
                        "angle_failure": bool(abs(obs[2]) > env.unwrapped.theta_threshold_radians),
                        "final_observation": obs.tolist(),
                        "last20": [{"step": steps[i], "observation": states[i].tolist(),
                                    "tree_action": actions[i], "teacher_action": labels[i]}
                                   for i in range(max(start, len(states)-20), len(states))]})
                    break
    finally:
        env.close()
    data = {"observations": np.asarray(states, dtype=np.float32),
            "teacher_actions": np.asarray(labels, dtype=np.int64),
            "executed_actions": np.asarray(actions, dtype=np.int64),
            "reset_seeds": np.asarray(seeds, dtype=np.int64), "steps": np.asarray(steps, dtype=np.int64)}
    return data, outcomes


def choose_round(rows):
    # Earliest round wins exact ties. Selection never reads test metrics.
    return max(rows, key=lambda r: (r["completion_rate"], r["mean_return"], -r["round"]))


def run(output):
    if output.exists():
        raise SystemExit(f"Already exists: {output}; supply a new --output directory")
    output.mkdir(parents=True)
    teacher_path = ROOT / "runs/ablation-mse-seed100/best.pt"
    baseline_path = ROOT / "runs/distillation-v1/selected_tree.json"
    dataset_path = ROOT / "runs/distillation-v1/demonstrations.npz"
    teacher = c.load_checkpoint(teacher_path).eval().requires_grad_(False)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    source = output / "source"; source.mkdir()
    for name in ["dagger.py", "distill.py", "cartpole.py", "model.py", "utils.py"]:
        (source / name).write_bytes((ROOT / name).read_bytes())
    d.write_json(output / "config.json", {"rounds": 5, "episodes_per_round": 50,
        "max_depth": 2, "tree_seed": 2026, "diagnostic_seeds": [10_000_000, 10_000_099],
        "collection_start_seeds": [11_000_000 + r*1000 for r in range(1,6)],
        "validation_seeds": [12_000_000, 12_000_099], "test_seeds": [13_000_000, 13_000_499],
        "selection": "highest validation completion, then mean return, then earliest round",
        "mixing": "pure learner rollouts (beta=0); uniform weight on all aggregated examples",
        "sklearn": sklearn.__version__, "numpy": np.__version__, "torch": str(torch.__version__),
        "gymnasium": c.gym.__version__,
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [teacher_path, baseline_path, dataset_path]}})
    diagnostic, outcomes = collect_labeled(teacher, baseline, 100, 10_000_000)
    np.savez_compressed(output / "diagnostic_states.npz", **diagnostic)
    failures = [o for o in outcomes if not o["completed"]]
    summary = {"episodes": 100, "failures": len(failures),
        "position_failures": sum(o["position_failure"] for o in outcomes),
        "angle_failures": sum(o["angle_failure"] for o in outcomes),
        "all_state_disagreement": float(np.mean(diagnostic["teacher_actions"] != diagnostic["executed_actions"])),
        "failure_tail_disagreement": float(np.mean([s["teacher_action"] != s["tree_action"] for o in failures for s in o["last20"]])) if failures else None}
    d.write_json(output / "failure_analysis.json", {"summary": summary, "episodes": outcomes})
    print(f"Baseline diagnosis: {summary}", flush=True)
    with np.load(dataset_path, allow_pickle=False) as original:
        train_mask = np.isin(original["episode_ids"], original["train_episode_ids"])
        x, y = original["observations"][train_mask].copy(), original["actions"][train_mask].copy()
        assert len(x) == 80_000
        assert not set(original["train_episode_ids"]) & set(original["validation_episode_ids"])
    base_validation = d.rollout(lambda obs: d.tree_action(baseline, obs), 100, 12_000_000)
    d.write_json(output / "baseline_validation.json", base_validation)
    rows, current = [], baseline
    for round_number in range(1, 6):
        batch, episodes = collect_labeled(teacher, current, 50, 11_000_000 + round_number*1000)
        np.savez_compressed(output / f"round{round_number}_new_examples.npz", **batch)
        d.write_json(output / f"round{round_number}_collection.json", episodes)
        x = np.concatenate([x, batch["observations"]]); y = np.concatenate([y, batch["teacher_actions"]])
        tree = DecisionTreeClassifier(max_depth=2, random_state=2026).fit(x, y)
        current = d.export_model(tree)
        assert current["leaves"] <= 4
        probes = x[::max(1,len(x)//2000)]
        np.testing.assert_array_equal(tree.predict(probes), [d.tree_action(current, obs) for obs in probes])
        d.write_json(output / f"round{round_number}_tree.json", current)
        (output / f"round{round_number}_rules.txt").write_text(export_text(tree, feature_names=d.FEATURES, decimals=8), encoding="utf-8")
        result = d.rollout(lambda obs: d.tree_action(current, obs), 100, 12_000_000)
        d.write_json(output / f"round{round_number}_validation.json", result)
        row = {"round": round_number, "leaves": current["leaves"], "added_examples": len(batch["teacher_actions"]),
               "total_examples": len(y), **{k:result[k] for k in ["mean_return","std_return","min_return","completion_rate"]}}
        rows.append(row)
        d.write_json(output / "rounds.json", rows)
        print(f"Round {round_number}: {row}", flush=True)
    selection = choose_round(rows)
    selected = json.loads((output / f"round{selection['round']}_tree.json").read_text())
    d.write_json(output / "selection.json", {"selected": selection, "baseline_validation": base_validation,
        "selected_before_test": True, "aspiration_completion": 0.9})
    d.write_json(output / "selected_tree.json", selected)
    (output / "selected_rules.txt").write_bytes((output / f"round{selection['round']}_rules.txt").read_bytes())
    tests = {}
    for name, action in [("baseline",lambda obs:d.tree_action(baseline,obs)),
                         ("aggregated",lambda obs:d.tree_action(selected,obs)),
                         ("teacher",lambda obs:c.select_action(teacher,obs))]:
        tests[name] = d.rollout(action, 500, 13_000_000)
        print(f"Fresh test {name}: mean {tests[name]['mean_return']:.2f}, completion {tests[name]['completion_rate']:.1%}", flush=True)
    d.write_json(output / "test_results.json", tests)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for ax,key,label in [(axes[0],"completion_rate","Completion rate"),(axes[1],"mean_return","Mean return")]:
        ax.plot(range(6), [base_validation[key]]+[r[key] for r in rows], "o-", color="#087e83")
        ax.set(xlabel="Aggregation round (0 = baseline)",ylabel=label,xticks=range(6))
    axes[0].axhline(.9,linestyle="--",color="gray",label="90% aspiration"); axes[0].set_ylim(0,1.05); axes[0].legend()
    axes[1].set_ylim(0,520); fig.suptitle("Fixed four-leaf budget · validation only"); fig.tight_layout()
    fig.savefig(output / "learning_curve.png",dpi=160); plt.close(fig)
    lines = ["# Dataset aggregation at a four-leaf budget", "", "## Failure diagnosis", "", json.dumps(summary),
        "", "## Validation rounds", "", "| Round | Added examples | Total examples | Leaves | Mean | Minimum | Completion |", "|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['round']} | {r['added_examples']} | {r['total_examples']} | {r['leaves']} | {r['mean_return']:.2f} | {r['min_return']:.0f} | {r['completion_rate']:.1%} |" for r in rows]
    lines += ["",f"Selected round {selection['round']} using validation only; baseline validation completion {base_validation['completion_rate']:.1%}.",
        "", "## Fresh test: 500 matched seeds per policy", "", "| Policy | Mean | Std | Minimum | Completion |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {name} | {r['mean_return']:.2f} | {r['std_return']:.2f} | {r['min_return']:.0f} | {r['completion_rate']:.1%} |" for name,r in tests.items()]
    lines += ["", "![Validation learning curve](learning_curve.png)", "", "## Scope", "",
        "One aggregation run and fitting seed; episode variation does not measure training-seed variability.",
        "Teacher labels were queried on learner states; learner actions were executed. Diagnostic, collection, validation, and test seed ranges are disjoint.",
        "Only the original 80,000 training examples were aggregated. The 20,000 original validation examples were excluded.",
        "All examples have equal weight. Later rounds accumulate the errors and distributions of earlier learners.",
        "Five rounds were pre-specified. No additional tuning or tree-size increase followed test evaluation.",
        "Disagreement near failures is descriptive, not proof that a particular action caused failure."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"runs/dagger-v1")
    args=parser.parse_args(); torch.set_num_threads(1); run(args.output)
