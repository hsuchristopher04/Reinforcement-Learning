"""Train and evaluate a Double DQN teacher on CartPole-v1."""
import argparse
import copy
import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from model import FC
from utils import ReplayBuffer

ENV_NAME = "CartPole-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_action(policy, observation, epsilon=0.0):
    if epsilon > 0 and np.random.random() < epsilon:
        return int(np.random.randint(policy.out_dim))
    with torch.no_grad():
        state = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE)
        return int(policy(state).argmax(dim=1).item())


def evaluate(policy, episodes, seed):
    """Use an independent environment and explicit episode seeds."""
    env = gym.make(ENV_NAME)
    was_training = policy.training
    policy.eval()
    returns = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            total = 0.0
            while True:
                observation, reward, terminated, truncated, _ = env.step(
                    select_action(policy, observation))
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
    finally:
        env.close()
        policy.train(was_training)
    return {"episodes": episodes, "seed": seed,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)), "returns": returns}


def train_policy(policy, target, buffer, optimizer, discount, loss_name="mse", bound_targets=False):
    state, action, next_state, reward, not_done = buffer.sample()
    current_q = policy(state).gather(1, action)
    with torch.no_grad():
        next_action = policy(next_state).argmax(dim=1, keepdim=True)
        expected_q = reward + discount * not_done * target(next_state).gather(1, next_action)
        if bound_targets:
            # CartPole has reward +1 each step. This is a task-specific bound,
            # not a generic rule for other environments or reward functions.
            expected_q.clamp_(0.0, 1.0 / (1.0 - discount))
    loss = (F.mse_loss if loss_name == "mse" else F.smooth_l1_loss)(current_q, expected_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def save_checkpoint(path, policy, step, seed):
    # Inference checkpoint, not an exact training-resumption snapshot.
    torch.save({"state_dict": policy.state_dict(), "env_name": ENV_NAME,
                "input_dim": policy.in_dim, "num_actions": policy.out_dim,
                "hidden_layers": policy.num_hidden_layers, "hidden_size": policy.layer_size,
                "step": step, "seed": seed}, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    if checkpoint["env_name"] != ENV_NAME:
        raise ValueError("Expected a CartPole-v1 teacher checkpoint")
    policy = FC(checkpoint["input_dim"], checkpoint["num_actions"],
                checkpoint["hidden_layers"], checkpoint["hidden_size"]).to(DEVICE)
    policy.load_state_dict(checkpoint["state_dict"])
    return policy


def plot_learning_curve(path, evaluations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([row[0] for row in evaluations], [row[1] for row in evaluations], marker="o")
    ax.set(xlabel="Training environment steps", ylabel="Mean validation return",
           title="CartPole Double DQN teacher", ylim=(0, 510))
    ax.axhline(475, linestyle="--", color="gray", label="Project target: 475")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train(args):
    seed_everything(args.seed)  # Seed before constructing networks.
    output = args.output or Path("runs") / datetime.now().strftime("cartpole-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(env_name=ENV_NAME, device=str(DEVICE), torch_version=str(torch.__version__),
                  gymnasium_version=gym.__version__, numpy_version=np.__version__)
    config.update(discount=0.99, batch_size=64, buffer_capacity=100_000,
                  target_tau=0.005, epsilon_start=1.0, epsilon_end=0.05,
                  epsilon_decay_steps=30_000, gradient_clip_norm=10.0)
    config["source_sha256"] = {}
    source_dir = output / "source"
    source_dir.mkdir()
    for name in ("cartpole.py", "model.py", "utils.py"):
        content = Path(__file__).with_name(name).read_bytes()
        (source_dir / name).write_bytes(content)
        config["source_sha256"][name] = hashlib.sha256(content).hexdigest()
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = gym.make(ENV_NAME)
    env.action_space.seed(args.seed)
    policy = FC(4, 2, 2, args.hidden_size).to(DEVICE)
    target = copy.deepcopy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    buffer = ReplayBuffer(4, 64, 100_000, DEVICE)
    observation, _ = env.reset(seed=args.seed)
    best_return = -float("inf")
    episode_return = 0.0
    episode_length = 0
    episode = 0
    evaluations = []
    print(f"Training on {DEVICE}; results: {output.resolve()}", flush=True)
    try:
        with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as ef, \
             (output / "evaluations.csv").open("w", newline="", encoding="utf-8") as vf:
            episodes_writer, evaluation_writer = csv.writer(ef), csv.writer(vf)
            episodes_writer.writerow(["step", "episode", "return", "length", "epsilon"])
            evaluation_writer.writerow(["step", "mean_return", "std_return"])
            for step in range(1, args.steps + 1):
                epsilon = 1.0 - 0.95 * min(1.0, max(0, step - args.warmup_steps) / 30_000)
                action = env.action_space.sample() if step <= args.warmup_steps else select_action(policy, observation, epsilon)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                # Bootstrap across time limits, but never across true terminal states.
                buffer.add(observation, action, next_observation, reward, terminated)
                observation = next_observation
                episode_return += reward
                episode_length += 1
                if step > args.warmup_steps and buffer.size >= 64 and step % args.train_every == 0:
                    train_policy(policy, target, buffer, optimizer, 0.99, args.loss, args.bound_targets)
                    with torch.no_grad():
                        for target_param, param in zip(target.parameters(), policy.parameters()):
                            if args.target_update == "soft":
                                target_param.lerp_(param, 0.005)
                if args.target_update == "hard" and step % args.target_every == 0:
                    target.load_state_dict(policy.state_dict())
                if terminated or truncated:
                    episode += 1
                    episodes_writer.writerow([step, episode, episode_return, episode_length, epsilon])
                    observation, _ = env.reset()
                    episode_return, episode_length = 0.0, 0
                if step % args.eval_every == 0 or step == args.steps:
                    result = evaluate(policy, args.eval_episodes, 1_000_000)
                    row = (step, result["mean_return"], result["std_return"])
                    evaluations.append(row)
                    evaluation_writer.writerow(row)
                    ef.flush()
                    vf.flush()
                    print(f"Step {step}: validation return {row[1]:.1f} +/- {row[2]:.1f}", flush=True)
                    if result["mean_return"] > best_return:
                        best_return = result["mean_return"]
                        save_checkpoint(output / "best.pt", policy, step, args.seed)
            save_checkpoint(output / "last.pt", policy, args.steps, args.seed)
    finally:
        env.close()
    if not args.skip_test:
        result = evaluate(load_checkpoint(output / "best.pt"), args.test_episodes, 2_000_000)
        (output / "test_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Best checkpoint held-out mean: {result['mean_return']:.1f} over {args.test_episodes} episodes", flush=True)
    plot_learning_curve(output / "learning_curve.png", evaluations)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--steps", type=positive_int, default=50_000)
    training.add_argument("--seed", type=int, default=100)
    training.add_argument("--output", type=Path)
    training.add_argument("--warmup-steps", type=positive_int, default=1000)
    training.add_argument("--eval-every", type=positive_int, default=5000)
    training.add_argument("--eval-episodes", type=positive_int, default=10)
    training.add_argument("--test-episodes", type=positive_int, default=100)
    training.add_argument("--learning-rate", type=positive_float, default=3e-4)
    training.add_argument("--loss", choices=["huber", "mse"], default="mse")
    training.add_argument("--bound-targets", action="store_true")
    training.add_argument("--hidden-size", type=positive_int, default=256)
    training.add_argument("--train-every", type=positive_int, default=1)
    training.add_argument("--target-update", choices=["soft", "hard"], default="soft")
    training.add_argument("--target-every", type=positive_int, default=1000)
    training.add_argument("--skip-test", action="store_true", help="Use validation only during tuning")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--episodes", type=positive_int, default=100)
    evaluation.add_argument("--seed", type=int, default=3_000_000)
    args = parser.parse_args()
    torch.set_num_threads(1)  # Small networks benefit from avoiding CPU thread overhead.
    if args.command == "train":
        train(args)
    else:
        print(json.dumps(evaluate(load_checkpoint(args.checkpoint), args.episodes, args.seed), indent=2))


if __name__ == "__main__":
    main()
