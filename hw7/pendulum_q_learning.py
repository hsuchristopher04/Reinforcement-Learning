import gymnasium as gym
import random
import numpy as np
from MDP import TD
from MDP import q_learning


def get_state_from_observation(obs):

    cos_theta = obs[0]
    sin_theta = obs[1]
    vel = obs[2]

    theta = np.arcsin(sin_theta)

    if sin_theta > 0 and cos_theta < 0:
        theta = np.pi - theta
        
    elif sin_theta < 0 and cos_theta < 0:
        theta = -np.pi - theta

    return (theta, vel)

def make_bins(low, high, n_bins):
    return np.linspace(low, high, n_bins + 1)[1:-1]

def discretize(obs, theta_bins, vel_bins):
    theta, vel = get_state_from_observation(obs)
    i = np.digitize(theta, theta_bins)
    j = np.digitize(vel, vel_bins)
    return i, j

def idx_from_state(i, j, n_i, n_j):
    return i * n_j + j

def argmax_rand(q_row, rng):
    m = q_row.max()
    idxs = np.flatnonzero(q_row == m)
    return rng.choice(idxs)

def eval_policy(env, q, actions, theta_bins, vel_bins, n_theta, n_vel,
                num_steps=200, num_episodes=100, seed=0):
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_ret = 0.0
        for _ in range(num_steps):
            i, j = discretize(obs, theta_bins, vel_bins)
            s = idx_from_state(i, j, n_theta, n_vel)
            a_idx = argmax_rand(q[s], rng)
            obs, r, terminated, truncated, _ = env.step(np.array([actions[a_idx]], dtype=np.float32))
            ep_ret += r
            if terminated or truncated:
                break
        total += ep_ret
    return total / num_episodes

if __name__ == '__main__':

    seed = 123
    rng = np.random.default_rng(seed)

    env = gym.make('Pendulum-v1')

    theta_min, theta_max = -np.pi, np.pi
    vel_min, vel_max = -8.0, 8.0
    act_min, act_max = env.action_space.low[0], env.action_space.high[0]

    # Assignment gamma
    gamma = 0.9

    # Discretization
    n_theta = 31
    n_vel = 31
    n_actions = 13
    actions = np.linspace(act_min, act_max, n_actions)

    theta_bins = make_bins(theta_min, theta_max, n_theta)
    vel_bins = make_bins(vel_min, vel_max, n_vel)

    n_states = n_theta * n_vel

    # Optimistic init + tiny noise
    q = 0.1 * np.ones((n_states, n_actions), dtype=np.float32) \
        + 0.005 * rng.standard_normal((n_states, n_actions)).astype(np.float32)

    # Training hyperparameters
    episodes = 6000
    max_steps = 200
    alpha_start, alpha_end = 0.30, 0.05
    eps_start, eps_end = 0.40, 0.02

    def schedule(v0, v1, t, T):
        return v1 + (v0 - v1) * max(0.0, (T - t) / max(1, T))

    target = -300.0
    eval_every = 50
    best_avg = -1e9
    consecutive_hits = 0

    print("Training Pendulum with Q-learning...")
    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        alpha = schedule(alpha_start, alpha_end, ep, episodes)
        eps = schedule(eps_start, eps_end, ep, episodes)

        for _ in range(max_steps):
            i, j = discretize(obs, theta_bins, vel_bins)
            s = idx_from_state(i, j, n_theta, n_vel)

            # epsilon-greedy
            if rng.random() < eps:
                a_idx = rng.integers(n_actions)
            else:
                a_idx = argmax_rand(q[s], rng)

            next_obs, r, terminated, truncated, _ = env.step(np.array([actions[a_idx]], dtype=np.float32))
            done = terminated or truncated

            i2, j2 = discretize(next_obs, theta_bins, vel_bins)
            s2 = idx_from_state(i2, j2, n_theta, n_vel)

            q_learning(q, alpha, gamma, s, a_idx, r, next_S=s2, last=done)

            obs = next_obs
            if done:
                break

        if ep % eval_every == 0:
            avg = eval_policy(env, q, actions, theta_bins, vel_bins, n_theta, n_vel,
                              num_steps=200, num_episodes=100,
                              seed=int(rng.integers(0, 2**31 - 1)))
            best_avg = max(best_avg, avg)
            print(f"[Pendulum-IMP] ep={ep:5d} eps={eps:.3f} alpha={alpha:.3f} avg_return={avg:.2f} best={best_avg:.2f}")

            # Lock-in behavior after hitting target: shrink eps/alpha and require 3 consecutive hits
            if avg >= target:
                consecutive_hits += 1
                eps_end = 0.01
                alpha_end = 0.02
            else:
                consecutive_hits = 0

            if consecutive_hits >= 3:
                print(f"[Pendulum-IMP] Target met consistently. Saving and stopping.")
                np.save("pendulum_q_table.npy", q)
                print("Saved Q to pendulum_q_table.npy")
                break

    # Final save + evaluation
    np.save("pendulum_q_table.npy", q)
    print("Saved Q to pendulum_q_table.npy")

    final_avg = eval_policy(env, q, actions, theta_bins, vel_bins, n_theta, n_vel,
                            num_steps=200, num_episodes=100,
                            seed=int(rng.integers(0, 2**31 - 1)))
    print(f"[Pendulum-IMP] Final average undiscounted return over 100 episodes: {final_avg:.2f}")