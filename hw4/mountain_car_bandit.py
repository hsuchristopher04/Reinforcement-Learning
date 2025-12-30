import gymnasium as gym
import random
import math
import itertools
import numpy as np


# hardcoded policy that achieves a reward of at least 95
# given it an observation, it returns an action
def policy_95(observation):
    pos, vel = observation
    if vel > 0 or pos > 0.2:
        action = .2
    else:
        action = -.2

    # make it a list/array for Gym
    return [action]

def discretize_state(observation):
    pos, vel = observation
    # Position is discretized based on being left or right of the valley bottom (-0.5)
    pos_bin = 0 if pos < -0.5 else 1
    # Velocity is discretized based on direction
    vel_bin = 0 if vel < 0 else 1
    # Combine them to get a single state index from 0 to 3
    return pos_bin * 2 + vel_bin

# generic policy function, where you can pass different controller pi
# given an observation, it returns the action selected by pi
def policy(observation, pi):
    state_index = discretize_state(observation)
    action = pi[state_index]
    return [action]


# this function runs an episode in Mountain Car, using policy pi, for the prescribed num_steps
def run_episode(env, pi, num_steps):
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(num_steps):
        # Use our generic policy function to get an action
        action = policy(obs, pi)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def ucb_search(env, num_steps, num_bandit_pulls, c):
    # Discretize the action space into 3 actions: strong left, neutral, strong right
    discrete_actions = [-0.2, 0.0, 0.2]
    num_discrete_actions = len(discrete_actions)
    
    # There are 4 discrete states, so a policy is a mapping from 4 states to 3 actions.
    # Total number of possible policies (arms) is 3^4 = 81.
    # We generate all possible policies using itertools.product.
    all_policies = list(itertools.product(discrete_actions, repeat=4))
    num_arms = len(all_policies)
    print(f"Generated {num_arms} unique policies (arms) to test.")

    # Initialize UCB variables
    arm_pulls = np.zeros(num_arms)
    arm_rewards = np.zeros(num_arms)
    
    print(f"\nStarting UCB search for {num_bandit_pulls} episodes...")
    # Main UCB loop
    for t in range(num_bandit_pulls):
        ucb_values = np.zeros(num_arms)
        
        # Initially, pull each arm once to get a baseline
        if t < num_arms:
            chosen_arm_index = t
        else:
            # Calculate UCB values for all arms
            for i in range(num_arms):
                if arm_pulls[i] > 0:
                    avg_reward = arm_rewards[i] / arm_pulls[i]
                    exploration_term = c * math.sqrt(math.log(t) / arm_pulls[i])
                    ucb_values[i] = avg_reward + exploration_term
                else:
                    # If an arm has never been pulled, give it max priority
                    ucb_values[i] = float('inf')
            
            # Choose the arm with the highest UCB value
            chosen_arm_index = np.argmax(ucb_values)
            
        # "Pull" the chosen arm by running an episode with its policy
        chosen_policy = all_policies[chosen_arm_index]
        reward = run_episode(env, chosen_policy, num_steps)
        
        # Update our knowledge of the arm
        arm_pulls[chosen_arm_index] += 1
        arm_rewards[chosen_arm_index] += reward

        if (t + 1) % 100 == 0:
            print(f"  UCB Episode {t+1}/{num_bandit_pulls} complete.")

    # Find the best arm based on average reward after the search
    arm_avg_rewards = np.divide(arm_rewards, arm_pulls, out=np.zeros_like(arm_rewards), where=arm_pulls!=0)
    best_arm_index = np.argmax(arm_avg_rewards)
    best_policy = all_policies[best_arm_index]
    
    print("\nUCB search complete.")
    print(f"Best policy found: {best_policy}")
    print(f"Expected reward from UCB: {arm_avg_rewards[best_arm_index]:.2f}")
    
    return best_policy



if __name__ == '__main__':

    np.random.seed(1)

    # Limit on the number of steps per episode in Mountain Car
    num_steps = 1000

    # use this if you would like to render the environment
    # env = gym.make('MountainCarContinuous-v0', render_mode = 'human').env.unwrapped

    # use this is you would like to run multiple runs quickly without rendering
    env = gym.make('MountainCarContinuous-v0').env.unwrapped

    # these variables may be useful in case you're unsure what the state-action space looks like
    pos_min = env.observation_space.low[0]
    pos_max = env.observation_space.high[0]

    vel_min = env.observation_space.low[1]
    vel_max = env.observation_space.high[1]

    act_min = env.action_space.low[0]
    act_max = env.action_space.high[0]

    # your code goes here
    total_reward = 0
    episodes = 100

    for _ in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0
        for _ in range(num_steps):
            action = policy_95(obs)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        total_reward += ep_reward

    avg_reward = total_reward / episodes
    print(f"policy_95: Average reward over {episodes} episodes: {avg_reward:.2f}")


    # --- UCB Execution ---
    # UCB Hyperparameters
    num_bandit_pulls = 800   # Total number of episodes to run for the search
    c = 10.0                 # Exploration constant. Higher 'c' encourages more exploration.
    
    # Run the UCB search to find the best policy
    best_policy_found = ucb_search(env, num_steps, num_bandit_pulls, c)
    
    # --- Final Evaluation ---
    # As required, evaluate the best policy found by UCB over 100 episodes
    print("\n--- Final Evaluation of Best Policy ---")
    total_reward_final = 0
    num_eval_episodes = 100
    
    for i in range(num_eval_episodes):
        reward = run_episode(env, best_policy_found, num_steps)
        total_reward_final += reward
        # print(f"  Evaluation Episode {i+1}: Reward = {reward:.2f}")

    avg_reward_final = total_reward_final / num_eval_episodes
    print(f"\nFinal Average Reward over {num_eval_episodes} episodes: {avg_reward_final:.2f}")

    env.close()

