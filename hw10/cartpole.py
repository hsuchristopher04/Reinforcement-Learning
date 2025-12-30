import gymnasium as gym
import numpy as np

from model import FC
import torch
from utils import ReplayBuffer
import copy
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(policy, state, num_actions, eps):
    """
    Epsilon-greedy action selection:
    - with prob eps: random action
    - with prob 1-eps: greedy w.r.t. Q-network
    """
    if np.random.rand() < eps:
        # random exploration
        return np.random.randint(num_actions)
    else:
        # greedy action from Q-network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, state_dim)
        with torch.no_grad():
            q_values = policy(state_tensor)  # (1, num_actions)
        action = q_values.argmax(dim=1).item()
        return action



def eval_policy(env_name, policy, num_actions, num_steps=1000, num_episodes=10):
    """
    Evaluate the current policy greedily (eps=0)
    and return average undiscounted reward.
    """
    env = gym.make(env_name)

    policy.eval()
    episode_returns = []

    with torch.no_grad():
        for ep in range(num_episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action = select_action(policy, obs, num_actions, eps=0.0) 
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1

                if steps >= num_steps:
                    break 

            episode_returns.append(total_reward)

    policy.train()
    avg_return = np.mean(episode_returns)
    print(f"[Eval] Avg return over {num_episodes} episodes: {avg_return:.2f}")
    return avg_return


def train_policy(policy, target_policy, buffer, optimizer, discount):
    """
    One DDQN update step:
    - sample batch from replay buffer
    - compute Double DQN target
    - MSE loss between Q(s,a) and target
    - gradient step
    """

    state, action, next_state, reward, not_done = buffer.sample()
    
    # Current Q(s, a)
    q_values = policy(state)
    current_q = q_values.gather(1, action)

    # Double DQN target: a* = argmax_a Q_policy(s', a)
    with torch.no_grad():
        q_next_policy = policy(next_state)
        next_actions = q_next_policy.argmax(dim=1, keepdim=True) 

        q_next_target = target_policy(next_state) 
        target_q_next = q_next_target.gather(1, next_actions)

        target = reward + discount * not_done * target_q_next

    # MSE loss
    loss = F.mse_loss(current_q, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':

    env_name = 'CartPole-v0'

    env = gym.make(env_name)

    # Hyperparameters -- feel free to change as you see fit
    batch_size = 64
    buffer_size = 1e6
    learning_rate = 3e-4
    eps = 0.1
    seed = 100
    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    num_hidden_layers = 2
    num_neurons_per_layer = 256
    discount = 0.99
    warmup_steps = 1e3
    target_update_freq = 1
    tau = 0.005

    replay_buffer = ReplayBuffer(input_dim, batch_size,	buffer_size, device)

    policy = FC(input_dim, num_actions, num_hidden_layers, num_neurons_per_layer).to(device)
    target_policy = copy.deepcopy(policy)

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    torch.manual_seed(seed)
    np.random.seed(seed)

    num_steps = 100000

    observation, _ = env.reset()

    episode_num = 1

    total_reward = 0

    for t in range(num_steps):

        # pre-collect some data using random actions
        if t < warmup_steps:
            action = env.action_space.sample()
        else:
            action = select_action(policy, observation, num_actions, eps)

        # make a step in the environment
        next_observation, reward, done, truncated, ___ = env.step(action)

        # add new experience to replay buffer
        replay_buffer.add(observation, action, next_observation, reward, done)
        observation = copy.copy(next_observation)

        total_reward += reward

        if t >= warmup_steps and replay_buffer.size >= batch_size:
            train_policy(policy, target_policy, replay_buffer, optimizer, discount)

            # update target network every target_update_freq steps
            if t % target_update_freq == 0:
                with torch.no_grad():
                    for param, target_param in zip(policy.parameters(), target_policy.parameters()):
                        target_param.data.copy_(
                            tau * param.data + (1.0 - tau) * target_param.data
                        )

        if (t + 1) % 10000 == 0:
            eval_policy(env_name, policy, num_actions)

        if done:
            print(f"Time Steps: {t}, Episode: {episode_num}, Reward: {total_reward}")
            observation, _ = env.reset()
            total_reward = 0
            episode_num += 1