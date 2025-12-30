import gymnasium as gym
import random
import numpy as np
from MDP import TD
from MDP import q_learning, TD as sarsa

def make_bins(low, high, n_bins):
    return np.linspace(low, high, n_bins + 1)[1:-1]

def discretize(obs, pos_bins, vel_bins):
    pos, vel = obs
    i = np.digitize(pos, pos_bins)
    j = np.digitize(vel, vel_bins)
    return i, j

def idx_from_state(i, j, n_i, n_j):
    return i * n_j + j

def argmax_rand(q_row, rng):
    max_val = np.max(q_row)
    candidates = np.flatnonzero(q_row == max_val)
    return rng.choice(candidates)

def eval_policy(env, q, actions, pos_bins, vel_bins, n_pos, n_vel,
                num_steps=1500, num_episodes=100, seed=0):
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_ret = 0.0
        for t in range(num_steps):
            i, j = discretize(obs, pos_bins, vel_bins)
            s = idx_from_state(i, j, n_pos, n_vel)
            a_idx = argmax_rand(q[s], rng)
            a = np.array([actions[a_idx]], dtype=np.float32)
            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            if terminated or truncated:
                break
        total += ep_ret
    return total / num_episodes

if __name__ == '__main__':
    seed = 42
    rng = np.random.default_rng(seed)

    env = gym.make('MountainCarContinuous-v0')

    pos_min = env.observation_space.low[0]
    pos_max = env.observation_space.high[0]

    vel_min = env.observation_space.low[1]
    vel_max = env.observation_space.high[1]

    act_min = env.action_space.low[0]
    act_max = env.action_space.high[0]

    # Hyperparameters
    gamma = 0.99
    n_pos = 25
    n_vel = 21
    n_actions = 15
    actions = np.linspace(act_min, act_max, n_actions)

    pos_bins = make_bins(pos_min, pos_max, n_pos)
    vel_bins = make_bins(vel_min, vel_max, n_vel)

    n_states = n_pos * n_vel

    # Optimistic init with tiny noise to break ties
    q = 0.5 * np.ones((n_states, n_actions), dtype=np.float32) \
        + 0.01 * rng.standard_normal((n_states, n_actions))

    # Training schedules
    episodes = 12000
    max_steps_train = 2000
    alpha_start = 0.35
    alpha_end = 0.05
    eps_start = 1.0
    eps_end = 0.05

    use_sarsa = False  # set True to try on-policy SARSA

    def schedule(v0, v1, t, T):
        return v1 + (v0 - v1) * max(0.0, (T - t) / max(1, T))

    target = 90.0
    eval_every = 100
    best_avg = -1e9

    print("Training MountainCarContinuous with Q-learning...")
    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        alpha = schedule(alpha_start, alpha_end, ep, episodes)
        eps = schedule(eps_start, eps_end, ep, episodes)

        # Initial state/action (needed for SARSA)
        i, j = discretize(obs, pos_bins, vel_bins)
        s = idx_from_state(i, j, n_pos, n_vel)
        if rng.random() < eps:
            a_idx = rng.integers(n_actions)
        else:
            a_idx = argmax_rand(q[s], rng)

        ep_ret = 0.0
        for t in range(max_steps_train):
            a = np.array([actions[a_idx]], dtype=np.float32)
            next_obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_ret += r

            i2, j2 = discretize(next_obs, pos_bins, vel_bins)
            s2 = idx_from_state(i2, j2, n_pos, n_vel)

            if use_sarsa:
                # pick next action for SARSA
                if rng.random() < eps:
                    a2_idx = rng.integers(n_actions)
                else:
                    a2_idx = argmax_rand(q[s2], rng)
                sarsa(q, alpha, gamma, s, a_idx, r, next_S=s2, next_A=a2_idx, last=done)
                a_idx = a2_idx
            else:
                # Q-learning update
                q_learning(q, alpha, gamma, s, a_idx, r, next_S=s2, last=done)
                # behavior: epsilon-greedy on the NEXT state
                if rng.random() < eps:
                    a_idx = rng.integers(n_actions)
                else:
                    a_idx = argmax_rand(q[s2], rng)

            s = s2
            if done:
                break

        if ep % eval_every == 0:
            avg = eval_policy(
                env, q, actions, pos_bins, vel_bins, n_pos, n_vel,
                num_steps=1500, num_episodes=100,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            best_avg = max(best_avg, avg)
            print(f"[MC-IMP] ep={ep:5d} eps={eps:.3f} alpha={alpha:.3f} avg_return={avg:.3f} best={best_avg:.3f}")
            if avg >= target:
                print(f"[MC-IMP] Target met (avg_return={avg:.2f}). Stopping early.")
                break

    final_avg = eval_policy(
        env, q, actions, pos_bins, vel_bins, n_pos, n_vel,
        num_steps=1500, num_episodes=100,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    print(f"[MC-IMP] Final average undiscounted return over 100 episodes: {final_avg:.3f}")