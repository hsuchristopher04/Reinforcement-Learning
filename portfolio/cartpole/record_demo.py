"""Record paired CartPole policies and build a standalone HTML playback demo."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import cartpole as c
import distill as d

ROOT = Path(__file__).resolve().parent


def decision_path(model, obs):
    state = np.asarray(obs, dtype=np.float32)
    node, path = 0, [0]
    while model["children_left"][node] != -1:
        left = float(state[model["feature"][node]]) <= model["threshold"][node]
        node = model["children_left" if left else "children_right"][node]
        path.append(node)
    return path


def record(policy, seed, model=None):
    env = c.gym.make(c.ENV_NAME)
    frames = []
    try:
        obs, _ = env.reset(seed=seed)
        total = 0.0
        for step in range(500):
            path = decision_path(model, obs) if model else []
            action = int(model["node_action"][path[-1]]) if model else c.select_action(policy, obs)
            frames.append({"step": step, "observation": obs.tolist(), "action": action,
                           "path": path, "status": "Running", "reason": "", "return": total})
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                reasons = []
                if abs(obs[0]) > env.unwrapped.x_threshold:
                    reasons.append("Cart crossed the track boundary")
                if abs(obs[2]) > env.unwrapped.theta_threshold_radians:
                    reasons.append("Pole exceeded the 12° limit")
                status = "Failed" if terminated else "Completed"
                frames.append({"step": step + 1, "observation": obs.tolist(), "action": None,
                               "path": [], "status": status,
                               "reason": "; ".join(reasons) if terminated else "Reached the 500-step limit",
                               "return": total})
                break
    finally:
        env.close()
    return {"frames": frames, "length": len(frames) - 1, "return": total,
            "status": frames[-1]["status"]}


def validate_episode(episode, model):
    teacher, tree = episode["teacher"], episode["tree"]
    assert teacher["frames"][0]["observation"] == tree["frames"][0]["observation"]
    for trajectory in [teacher, tree]:
        assert len(trajectory["frames"]) == trajectory["length"] + 1
        assert trajectory["frames"][-1]["action"] is None
        for i, frame in enumerate(trajectory["frames"]):
            assert frame["step"] == i
            assert frame["return"] == i
    for frame in tree["frames"][:-1]:
        assert frame["action"] == d.tree_action(model, frame["observation"])
        assert frame["path"] == decision_path(model, frame["observation"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=ROOT / "runs/ablation-mse-seed100/best.pt")
    parser.add_argument("--tree", type=Path, default=ROOT / "runs/distillation-v1/selected_tree.json")
    parser.add_argument("--output", type=Path, default=ROOT / "demo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[9_000_001, 9_000_000])
    args = parser.parse_args()
    torch.set_num_threads(1)
    teacher = c.load_checkpoint(args.teacher).eval()
    teacher.requires_grad_(False)
    model = json.loads(args.tree.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = []
    for seed in args.seeds:
        episode = {"seed": seed, "teacher": record(teacher, seed), "tree": record(None, seed, model)}
        validate_episode(episode, model)
        episode["label"] = "Both complete" if episode["teacher"]["status"] == episode["tree"]["status"] == "Completed" else "Policy comparison"
        if episode["teacher"]["status"] == "Completed" and episode["tree"]["status"] == "Failed":
            episode["label"] = "Tree fails, teacher completes"
        episodes.append(episode)
        print(f"Seed {seed}: teacher {episode['teacher']['return']}, tree {episode['tree']['return']}", flush=True)
    # Independently reproduce aggregate results; example selection does not alter them.
    metrics = {"teacher": d.rollout(lambda obs: c.select_action(teacher, obs), 100, 9_000_000),
               "tree": d.rollout(lambda obs: d.tree_action(model, obs), 100, 9_000_000)}
    data = {"env": c.ENV_NAME, "dt": 0.02, "tree_model": model, "episodes": episodes,
            "aggregate": metrics,
            "provenance": {"teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
                           "tree_sha256": hashlib.sha256(args.tree.read_bytes()).hexdigest(),
                           "gymnasium": c.gym.__version__, "torch": str(torch.__version__),
                           "numpy": np.__version__}}
    payload = json.dumps(data, separators=(",", ":"))
    (args.output / "trajectories.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    template = (ROOT / "demo_template.html").read_text(encoding="utf-8")
    (args.output / "index.html").write_text(template.replace("__DEMO_DATA__", payload.replace("<", "\\u003c")), encoding="utf-8")
    print(f"Demo: {args.output / 'index.html'}", flush=True)
    print(f"Tree aggregate: {metrics['tree']['mean_return']}, completion {metrics['tree']['completion_rate']}", flush=True)


if __name__ == "__main__":
    main()
