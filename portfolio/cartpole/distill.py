"""Collect teacher demonstrations, fit decision trees, and evaluate control."""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import sklearn
import torch
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import cartpole as c

ROOT = Path(__file__).resolve().parent
FEATURES = ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"]
DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def episode_split(episode_ids, seed=2026):
    ids = np.unique(episode_ids)
    if len(ids) < 5:
        raise ValueError("At least five episodes are required")
    shuffled = np.random.default_rng(seed).permutation(ids)
    count = max(1, round(len(ids) * 0.2))
    validation_ids = np.sort(shuffled[:count])
    training_ids = np.sort(shuffled[count:])
    return training_ids, validation_ids


def collect(teacher, episodes, seed):
    observations, actions, episode_ids, reset_seeds, returns = [], [], [], [], []
    env = gym.make(c.ENV_NAME)
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total = 0.0
            while True:
                action = c.select_action(teacher, obs)
                observations.append(obs.copy())
                actions.append(action)
                episode_ids.append(episode)
                reset_seeds.append(seed + episode)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
            if (episode + 1) % 20 == 0 or episode + 1 == episodes:
                print(f"Collected {episode + 1}/{episodes} episodes ({len(actions)} examples)", flush=True)
    finally:
        env.close()
    return dict(observations=np.asarray(observations, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                episode_ids=np.asarray(episode_ids, dtype=np.int64),
                reset_seeds=np.asarray(reset_seeds, dtype=np.int64),
                episode_returns=np.asarray(returns, dtype=np.float32))


def export_model(tree):
    structure = tree.tree_
    return {"format_version": 1, "env_name": c.ENV_NAME, "features": FEATURES,
            "actions": {"0": "left", "1": "right"},
            "depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves()),
            "children_left": structure.children_left.tolist(),
            "children_right": structure.children_right.tolist(),
            "feature": structure.feature.tolist(), "threshold": structure.threshold.tolist(),
            "node_action": tree.classes_[structure.value[:, 0, :].argmax(axis=1)].astype(int).tolist()}


def tree_action(model, observation):
    # sklearn converts input to float32; preserve identical threshold semantics.
    observation = np.asarray(observation, dtype=np.float32)
    node = 0
    visited = 0
    while model["children_left"][node] != -1:
        visited += 1
        if visited > len(model["children_left"]):
            raise ValueError("Invalid tree: cyclic child references")
        # Promote float32 input back to Python float before comparing with the
        # float64 threshold; NumPy scalar promotion otherwise rounds the threshold.
        branch = "children_left" if float(observation[model["feature"][node]]) <= model["threshold"][node] else "children_right"
        node = model[branch][node]
        if not isinstance(node, int) or not 0 <= node < len(model["children_left"]):
            raise ValueError("Invalid tree: child index out of bounds")
    return int(model["node_action"][node])


def rollout(action_fn, episodes, seed):
    env = gym.make(c.ENV_NAME)
    returns, successes = [], 0
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total, length = 0.0, 0
            while True:
                obs, reward, terminated, truncated, _ = env.step(action_fn(obs))
                total += reward
                length += 1
                if terminated or truncated:
                    successes += int(length == 500 and truncated and not terminated)
                    break
            returns.append(total)
    finally:
        env.close()
    return {"episodes": episodes, "seed": seed, "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)), "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)), "completion_rate": successes / episodes,
            "returns": returns}


def select_candidate(rows, threshold=475):
    eligible = [row for row in rows if row["mean_return"] >= threshold]
    if eligible:
        return min(eligible, key=lambda r: (r["leaves"], r["actual_depth"], -r["mean_return"])), True
    return min(rows, key=lambda r: (-r["mean_return"], r["leaves"], r["actual_depth"])), False


def plots(output, rows, teacher_mean, selected_tree):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["leaves"] for r in rows], [r["mean_return"] for r in rows], "o-")
    for row in rows:
        ax.annotate(f"d={row['max_depth']}", (row["leaves"], row["mean_return"]),
                    xytext=(4, 6), textcoords="offset points", fontsize=8)
    ax.axhline(teacher_mean, color="black", linestyle=":", label="Teacher validation mean")
    ax.axhline(475, color="gray", linestyle="--", label="Selection target")
    ax.set(xscale="log", xlabel="Number of leaves (log scale)", ylabel="Mean validation return",
           title="Decision-tree complexity versus control performance", ylim=(0, 540))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "complexity_vs_return.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(selected_tree, feature_names=FEATURES, class_names=["left", "right"],
              max_depth=3, filled=True, impurity=False, fontsize=7, ax=ax)
    ax.set_title("Selected policy: first four levels; (...) marks omitted subtrees")
    fig.tight_layout()
    fig.savefig(output / "selected_tree_overview.png", dpi=160)
    plt.close(fig)


def run(args):
    output = args.output or ROOT / "runs" / datetime.now().strftime("distillation-%Y%m%d-%H%M%S-%f")
    if output.exists():
        raise SystemExit(f"Output already exists: {output}. Omit --output for a new timestamped folder.")
    teacher = c.load_checkpoint(args.teacher)
    teacher.eval()
    teacher.requires_grad_(False)
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    hashes = {}
    for name in ["distill.py", "cartpole.py", "model.py", "utils.py"]:
        data = (ROOT / name).read_bytes()
        (source / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    write_json(output / "config.json", {"teacher": str(args.teacher.resolve()),
        "teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
        "source_sha256": hashes, "env_name": c.ENV_NAME, "depths": DEPTHS,
        "collection_episodes": args.episodes, "collection_seed": 6_000_000,
        "split_seed": 2026, "tree_seed": 2026, "control_validation_seed": 7_000_000,
        "test_seed": 8_000_000, "evaluation_episodes": args.eval_episodes,
        "selection_target": 475, "sklearn": sklearn.__version__, "numpy": np.__version__,
        "torch": str(torch.__version__), "gymnasium": gym.__version__})
    print(f"Output: {output.resolve()}", flush=True)
    if args.demonstrations:
        previous = json.loads(args.demonstrations.with_name("config.json").read_text(encoding="utf-8"))
        if previous["teacher_sha256"] != hashlib.sha256(args.teacher.read_bytes()).hexdigest():
            raise ValueError("Demonstrations were collected from a different teacher")
        if previous["collection_episodes"] != args.episodes or previous["collection_seed"] != 6_000_000:
            raise ValueError("Demonstration count or seed range does not match this run")
        with np.load(args.demonstrations, allow_pickle=False) as data:
            dataset = {key: data[key].copy() for key in data.files}
        print(f"Reusing demonstrations from {args.demonstrations}", flush=True)
    else:
        dataset = collect(teacher, args.episodes, 6_000_000)
    train_ids, val_ids = episode_split(dataset["episode_ids"])
    dataset.update(train_episode_ids=train_ids, validation_episode_ids=val_ids)
    np.savez_compressed(output / "demonstrations.npz", **dataset)
    train_mask = np.isin(dataset["episode_ids"], train_ids)
    val_mask = np.isin(dataset["episode_ids"], val_ids)
    x, y = dataset["observations"], dataset["actions"]
    rows, trees, exported, detailed = [], {}, {}, {}
    teacher_validation = rollout(lambda obs: c.select_action(teacher, obs), args.eval_episodes, 7_000_000)
    print(f"Teacher validation: {teacher_validation['mean_return']:.2f}", flush=True)
    for depth in DEPTHS:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=2026)
        tree.fit(x[train_mask], y[train_mask])
        model = export_model(tree)
        # Verify the exact JSON artifact that will control the environment.
        model = json.loads(json.dumps(model))
        probe = x[val_mask][::max(1, int(val_mask.sum()) // 1000)]
        np.testing.assert_array_equal(tree.predict(probe), [tree_action(model, obs) for obs in probe])
        write_json(output / f"tree_depth{depth}.json", model)
        (output / f"tree_depth{depth}_rules.txt").write_text(
            export_text(tree, feature_names=FEATURES, max_depth=20, decimals=6), encoding="utf-8")
        result = rollout(lambda obs: tree_action(model, obs), args.eval_episodes, 7_000_000)
        row = {"max_depth": depth, "actual_depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves()),
               "action_accuracy": float(tree.score(x[val_mask], y[val_mask])),
               **{k: result[k] for k in ["mean_return", "std_return", "completion_rate"]}}
        rows.append(row)
        trees[depth], exported[depth], detailed[str(depth)] = tree, model, result
        print(f"Depth {depth}: {row['leaves']} leaves, agreement {row['action_accuracy']:.2%}, "
              f"return {row['mean_return']:.2f}, completion {row['completion_rate']:.0%}", flush=True)
    with (output / "validation_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "validation_rollouts.json", {"teacher": teacher_validation, "trees": detailed})
    selected, qualified = select_candidate(rows)
    depth = selected["max_depth"]
    # Persist the selection BEFORE examining any test outcomes.
    write_json(output / "selection.json", {"selected": selected, "qualified_on_validation": qualified,
        "rule": "fewest leaves with mean >=475; otherwise highest mean (exploratory fallback)",
        "test_seed": 8_000_000})
    write_json(output / "selected_tree.json", exported[depth])
    (output / "selected_tree_rules.txt").write_text(
        export_text(trees[depth], feature_names=FEATURES, max_depth=20, decimals=6), encoding="utf-8")
    teacher_test = rollout(lambda obs: c.select_action(teacher, obs), args.eval_episodes, 8_000_000)
    tree_test = rollout(lambda obs: tree_action(exported[depth], obs), args.eval_episodes, 8_000_000)
    write_json(output / "test_results.json", {"teacher": teacher_test, "tree": tree_test})
    plots(output, rows, teacher_validation["mean_return"], trees[depth])
    lines = ["# Decision-tree distillation results", "",
        f"Collected {len(y):,} demonstrations from {args.episodes} frozen-teacher episodes.",
        f"Split: {len(train_ids)} training episodes ({train_mask.sum():,} rows), "
        f"{len(val_ids)} action-validation episodes ({val_mask.sum():,} rows).", "",
        "## Control validation", "",
        "All candidates used the same fresh reset seeds, separate from demonstrations.", "",
        "| Maximum depth | Leaves | Action agreement | Mean return | Return std | Completion |",
        "|---|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {r['max_depth']} | {r['leaves']} | {r['action_accuracy']:.2%} | "
                 f"{r['mean_return']:.2f} | {r['std_return']:.2f} | {r['completion_rate']:.0%} |" for r in rows)
    lines += ["", f"Selected depth {depth}, with {selected['leaves']} leaves.",
        "The selected tree met the validation target." if qualified else
        "No tree met the validation target; the selected tree is the best exploratory baseline.",
        "", "## Fresh test (selection frozen beforehand)", "",
        f"Each policy used {args.eval_episodes} episodes starting at seed 8,000,000.",
        f"Teacher: mean {teacher_test['mean_return']:.2f}, std {teacher_test['std_return']:.2f}, "
        f"completion {teacher_test['completion_rate']:.0%}.",
        f"Tree: mean {tree_test['mean_return']:.2f}, std {tree_test['std_return']:.2f}, "
        f"completion {tree_test['completion_rate']:.0%}.", "",
        "Completion requires reaching the 500-step time limit without physical failure.",
        "Action agreement is measured on teacher-visited states; return measures the tree's own control.",
        "One demonstration collection and one tree-fitting seed were used; uncertainty across datasets",
        "is not measured. Return std describes episodes, not independent training runs.", "",
        "![Complexity versus return](complexity_vs_return.png)", "",
        "![Selected tree overview](selected_tree_overview.png)", "",
        "The overview truncates deeper branches. Full rules are in selected_tree_rules.txt."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Selected depth {depth}; target met on validation: {qualified}. "
          f"Fresh test: tree {tree_test['mean_return']:.2f}, teacher {teacher_test['mean_return']:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("run")
    training.add_argument("--teacher", type=Path, default=ROOT / "runs/ablation-mse-seed100/best.pt")
    training.add_argument("--output", type=Path)
    training.add_argument("--demonstrations", type=Path, help="Reuse an NPZ dataset with its sibling config.json")
    training.add_argument("--episodes", type=c.positive_int, default=200)
    training.add_argument("--eval-episodes", type=c.positive_int, default=100)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("tree", type=Path)
    evaluation.add_argument("--episodes", type=c.positive_int, default=100)
    evaluation.add_argument("--seed", type=int, default=9_000_000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "run":
        if not 5 <= args.episodes <= 1_000_000:
            parser.error("--episodes must be between 5 and 1,000,000 to preserve disjoint seed ranges")
        if args.eval_episodes > 1_000_000:
            parser.error("--eval-episodes must not exceed 1,000,000")
        run(args)
    else:
        model = json.loads(args.tree.read_text(encoding="utf-8"))
        if model["env_name"] != c.ENV_NAME or model["format_version"] != 1:
            parser.error("Unsupported tree model")
        print(json.dumps(rollout(lambda obs: tree_action(model, obs), args.episodes, args.seed), indent=2))


if __name__ == "__main__":
    main()
