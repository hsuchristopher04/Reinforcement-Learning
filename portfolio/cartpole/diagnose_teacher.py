"""Inspect greedy teacher failures and Q estimates on validation seeds."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import cartpole as c


def diagnose(checkpoint, episodes=30, seed=1_000_000):
    policy = c.load_checkpoint(checkpoint)
    policy.eval()
    env = c.gym.make(c.ENV_NAME)
    counts = {"position_failures": 0, "angle_failures": 0, "time_limits": 0}
    returns, values = [], []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            total = 0.0
            while True:
                with torch.no_grad():
                    q = policy(torch.as_tensor(obs, dtype=torch.float32, device=c.DEVICE))
                values.append(float(q.max().item()))
                obs, reward, terminated, truncated, _ = env.step(int(q.argmax().item()))
                total += reward
                if terminated or truncated:
                    counts["position_failures"] += int(abs(obs[0]) > env.unwrapped.x_threshold)
                    counts["angle_failures"] += int(abs(obs[2]) > env.unwrapped.theta_threshold_radians)
                    counts["time_limits"] += int(truncated)
                    break
            returns.append(total)
    finally:
        env.close()
    return {"checkpoint": str(checkpoint), "episodes": episodes, "seed": seed,
            "mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
            "mean_selected_q": float(np.mean(values)), "max_selected_q": max(values),
            "discounted_return_upper_bound": 100.0, **counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--episodes", type=c.positive_int, default=30)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = [diagnose(path, args.episodes, args.seed) for path in args.checkpoints]
    document = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(document, encoding="utf-8")
    print(document)
