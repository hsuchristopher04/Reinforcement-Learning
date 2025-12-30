import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from MDP import get_MDP
from MDP import policy_iteration
from MDP import value_iteration
from six.moves import cPickle as pickle
import sys

def get_state_from_observation(obs):
    cos_theta = obs[0]
    sin_theta = obs[1]
    vel = obs[2]

    # Your original mapping with quadrant correction
    theta = np.arcsin(sin_theta)
    if sin_theta > 0 and cos_theta < 0:
        theta = np.pi - theta
    elif sin_theta < 0 and cos_theta < 0:
        theta = -np.pi - theta

    return (theta, vel)

# ---- local helpers for mapping obs -> discrete state using saved MDP['edges'] ----
def _digitize_scalar(x, edges):
    i = np.searchsorted(edges, x, side='right') - 1
    return int(np.clip(i, 0, len(edges) - 2))

def _obs_to_state_tuple(obs_raw, edges):
    theta, thd = get_state_from_observation(obs_raw)
    return (_digitize_scalar(float(theta), edges[0]),
            _digitize_scalar(float(thd),   edges[1]))

if __name__ == '__main__':

    np.random.seed(1)

    num_data_per_state = 100
    env = gym.make('Pendulum-v1').env.unwrapped

    theta_min = -np.pi
    theta_max = np.pi

    vel_min = env.observation_space.low[2]   # typically -8
    vel_max = env.observation_space.high[2]  # typically  8

    act_min = env.action_space.low[0]        # -2
    act_max = env.action_space.high[0]       #  2

    # if you don't want to rebuild your MDP every time, you
    # can save/load it as a pickle, for example
    # your submitted code should also build the MDP and run for less than 10 minutes total
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        with open(filename, 'rb') as f:
            Pendulum_MDP = pickle.load(f)

    else:
        # ---- Build the tabular MDP for Pendulum ----
        # Discretization and actions you can tune:
        THETA_BINS = 41
        VEL_BINS   = 41
        ACTIONS    = [-2.0, -1.0, 0.0, 1.0, 2.0]

        Pendulum_MDP = get_MDP(
            env=env,
            MDP_state_bounds=[(theta_min, theta_max, THETA_BINS),
                              (vel_min,   vel_max,   VEL_BINS)],
            action_space=ACTIONS,
            num_data_per_state=num_data_per_state,
            obs=get_state_from_observation  # maps raw obs -> (theta, theta_dot)
        )

        filename = 'pendulum_mdp.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(Pendulum_MDP, f, pickle.HIGHEST_PROTOCOL)

    train_gamma = 0.9   # for planning (discounted, infinite-horizon)
    sim_gamma   = 1.0   # evaluation uses undiscounted episodic sum
    T = 200
    num_episodes = 100

    # ----------------------------------------------------------------
    # PLAN: Value Iteration (stationary policy for continuing MDP)
    # ----------------------------------------------------------------
    # Either PI or VI is fine; VI is usually simpler & robust here.
    V, pi = value_iteration(Pendulum_MDP, action_space_env=Pendulum_MDP['actions'],
                            gamma=train_gamma, eps=1e-3)

    # ----------------------------------------------------------------
    # Evaluate policy on the true simulator (undiscounted return)
    # ----------------------------------------------------------------
    returns = []
    for __ in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        all_rewards = []

        for _t in range(T):
            s_tuple = _obs_to_state_tuple(obs, Pendulum_MDP['edges'])
            a_idx = pi.get(s_tuple, 2)  # default to zero torque if unseen; index 2 -> 0.0 in ACTIONS
            a_scalar = Pendulum_MDP['actions'][a_idx]
            action = np.array([a_scalar], dtype=np.float32)

            obs, r, terminated, truncated, _ = env.step(action)
            total_reward += float(r)
            all_rewards.append(float(r))
            # Pendulum typically doesn't terminate early; we run fixed T steps.

        returns.append(total_reward)

    mean_R = float(np.mean(returns))
    std_R  = float(np.std(returns))
    print(f"[Pendulum VI] Average undiscounted return over {num_episodes} eps × {T} steps: {mean_R:.2f} ± {std_R:.2f}")

    # Optional histogram
    try:
        plt.figure()
        plt.hist(returns, bins=15)
        plt.xlabel("Episode return")
        plt.ylabel("Count")
        plt.title("Pendulum-v1: Value Iteration policy returns")
        plt.show()
    except Exception:
        pass
