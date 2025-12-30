import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from MDP import get_MDP
from MDP import dynamic_programming_finite_horizon
import sys
import pickle

# --- local helpers (state digitization to match your saved MDP grid) ---
def _digitize_scalar(x, edges):
    i = np.searchsorted(edges, x, side='right') - 1
    return int(np.clip(i, 0, len(edges) - 2))

def _obs_to_state_tuple(obs, edges):
    # obs = [position, velocity]
    return (_digitize_scalar(float(obs[0]), edges[0]),
            _digitize_scalar(float(obs[1]), edges[1]))

if __name__ == '__main__':

    np.random.seed(1)

    num_data_per_state = 120
    env = gym.make('MountainCarContinuous-v0').env.unwrapped

    pos_min = env.observation_space.low[0]
    pos_max = env.observation_space.high[0]

    vel_min = env.observation_space.low[1]
    vel_max = env.observation_space.high[1]

    act_min = env.action_space.low[0]
    act_max = env.action_space.high[0]

    # if you don't want to rebuild your MDP every time, you
    # can save/load it as a pickle, for example
    # your submitted code should also build the MDP and run for less than 10 minutes total
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        with open(filename, 'rb') as f:
            MC_MDP = pickle.load(f)

    else:
        # --- Build the tabular MDP by sampling inside each discretized state cell ---
        # Decent baseline grid; you can tweak to 25x35 if you need a bit more fidelity.
        POS_BINS = 31
        VEL_BINS = 41
        # Three discrete thrusts in [-1, 1]. You can add -0.5, 0.5 if needed.
        ACTIONS = [-1.0, -0.6, 0.6, 1.0]

        MC_MDP = get_MDP(
            env=env,
            MDP_state_bounds=[(pos_min, pos_max, POS_BINS),
                              (vel_min, vel_max, VEL_BINS)],
            action_space=ACTIONS,
            num_data_per_state=num_data_per_state,
            obs=None  # raw obs already [pos, vel]
        )

        filename = 'mc_mdp.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(MC_MDP, f, pickle.HIGHEST_PROTOCOL)

    gamma = 1.0
    T = 150
    num_episodes = 100

    # ----------------------------
    # BACKWARD INDUCTION (finite-H)
    # ----------------------------
    # action_space_env: env expects Box(1,), but our DP uses indexes, so we just pass the scalar list
    # (dynamic_programming_finite_horizon ignores this in the MDP.py I provided, but pass it for consistency)
    V_list, Pi_list = dynamic_programming_finite_horizon(
        MDP=MC_MDP,
        action_space_env=MC_MDP['actions'],
        gamma=gamma,
        T=T
    )

    # ----------------------------
    # Evaluate the time-dependent policy on the real env
    # ----------------------------
    returns = []
    for __ in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        all_rewards = []

        for t in range(T):
            s_tuple = _obs_to_state_tuple(obs, MC_MDP['edges'])
            # default to neutral thrust if state was never seen
            a_idx = Pi_list[t].get(s_tuple, 1)
            a_scalar = MC_MDP['actions'][a_idx]
            action = np.array([a_scalar], dtype=np.float32)

            obs, r, terminated, truncated, _ = env.step(action)
            total_reward += float(r)
            all_rewards.append(float(r))
            if terminated or truncated:
                break

        returns.append(total_reward)

    mean_R = float(np.mean(returns))
    std_R = float(np.std(returns))
    print(f"[MC DP] Average undiscounted return over {num_episodes} eps: {mean_R:.2f} ± {std_R:.2f}")

    # success rate
    successes = 0
    steps_to_goal = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        total = 0.0
        for t in range(T):
            s = _obs_to_state_tuple(obs, MC_MDP['edges'])
            a_idx = Pi_list[t].get(s, 0)  # default to a strong action
            action = np.array([MC_MDP['actions'][a_idx]], dtype=np.float32)
            obs, r, done, trunc, _ = env.step(action)
            total += r
            if float(obs[0]) >= env.unwrapped.goal_position:
                successes += 1
                steps_to_goal.append(t+1)
                break
    # print:
    print(f"[MC] success rate: {successes}/{num_episodes} = {successes/num_episodes:.1%}")
    if steps_to_goal:
        print(f"[MC] median steps to goal: {np.median(steps_to_goal):.0f}")


    # Optional: quick plot of return distribution
    try:
        plt.figure()
        plt.hist(returns, bins=15)
        plt.xlabel("Episode return")
        plt.ylabel("Count")
        plt.title("MountainCarContinuous-v0: finite-horizon DP policy returns")
        plt.show()
    except Exception:
        pass
