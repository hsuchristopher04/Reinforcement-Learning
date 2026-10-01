import gymnasium as gym
from ale_py import ALEInterface
import numpy as np
import cv2
import os

from model import FC
import torch
from utils import ReplayBuffer
import copy
import torch.optim as optim
import torch.nn.functional as F

LEFT_PADDLE_COLUMN = 16
RIGHT_PADDLE_COLUMN = 142
PADDLE_LENGTH = 16

BACKGROUND_COLOR1 = 64
BACKGROUND_COLOR2 = 95
LEFT_PADDLE_COLOR = 123
RIGHT_PADDLE_COLOR = 147
BALL_COLOR = 236


def env_step(env, action, frame_skip):

    total_reward = 0

    for f in range(frame_skip):
        state, reward, done, _, _ = env.step(action)
        total_reward += reward

        if reward != 0:
            break

        if done:
            break

    return state, total_reward, done


def get_paddle_position(paddle_col, image_size_y, color):
    temp = paddle_col

    # ball also in this column
    if np.max(temp) == BALL_COLOR:

        temp[temp == BALL_COLOR] = 0

    num_paddle_pixels = np.sum(temp == color)
    if num_paddle_pixels == PADDLE_LENGTH:
        pos_pixel = np.argmax(temp) + PADDLE_LENGTH / 2
    else:
        top_pixel = np.argmax(temp)
        if top_pixel == 0:
            pos_pixel = num_paddle_pixels - PADDLE_LENGTH / 2
        else:
            pos_pixel = top_pixel + PADDLE_LENGTH / 2

    return pos_pixel / image_size_y


def get_ball_pos(image, image_size_x, image_size_y):
    # handle the case of no ball at the beginning of each round
    ball_pos = (-1, -1)
    for col in range(0, image_size_x):
        max_ind = np.argmax(image[:, col])

        if image[max_ind, col] == BALL_COLOR:
            ball_pos = (col / image_size_x, max_ind / image_size_y)

    return ball_pos


def get_state_from_image(image):

    # convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get rid of borders up and down
    cropped_image = gray_image[34:194, :]
    image_size_x = cropped_image.shape[0]
    image_size_y = cropped_image.shape[1]

    # handle first few frames with no left paddle
    if (np.max(cropped_image[:, LEFT_PADDLE_COLUMN]) == BACKGROUND_COLOR1 or
            np.max(cropped_image[:, LEFT_PADDLE_COLUMN]) == BACKGROUND_COLOR2):
        left_pos = -1
    else:
        left_pos = get_paddle_position(cropped_image[:, LEFT_PADDLE_COLUMN], image_size_y, LEFT_PADDLE_COLOR)

    right_pos = get_paddle_position(cropped_image[:, RIGHT_PADDLE_COLUMN], image_size_y, RIGHT_PADDLE_COLOR)

    ball_pos = get_ball_pos(cropped_image, image_size_x, image_size_y)

    return np.array([left_pos, right_pos, ball_pos[0], ball_pos[1]])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(policy, state, num_actions, eps):
    # epsilon-greedy action selection
    if np.random.rand() < eps:
        # explore
        return np.random.randint(num_actions)
    else:
        # exploit: greedy action from Q-network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy(state_tensor)
        action = q_values.argmax(dim=1).item()
        return action


def eval_policy(env_name, policy, num_actions, frame_skip, num_steps=1000, num_episodes=10):
    """
    Evaluate the current policy on full Pong games (to 21 points).
    Uses greedy actions (eps=0). Returns average undiscounted return
    over num_episodes games.
    """
    env = gym.make(env_name)

    policy.eval()
    episode_returns = []

    with torch.no_grad():
        for ep in range(num_episodes):
            obs, _ = env.reset()
            state = get_state_from_image(obs)
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = select_action(policy, state, num_actions, eps=0.0)

                next_obs, reward, done = env_step(env, action, frame_skip)
                next_state = get_state_from_image(next_obs)

                state = next_state
                total_reward += reward
                steps += 1

                if steps >= num_steps:
                    # safety cap if something weird happens; usually not hit
                    break

            episode_returns.append(total_reward)

    policy.train()
    avg_return = np.mean(episode_returns)
    print(f"[Pong Eval] Avg return over {num_episodes} full games: {avg_return:.2f}")
    return avg_return



def train_policy(policy, target_policy, buffer, optimizer, discount):

    # sample a batch from replay buffer
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

    loss = F.mse_loss(current_q, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    pass

if __name__ == '__main__':

    env_name = 'PongNoFrameskip-v4'

    env = gym.make(env_name)

    # Hyperparameters -- feel free to change as you see fit
    batch_size = 64
    buffer_size = 1e6
    learning_rate = 3e-4
    eps = 0.1
    seed = 100
    input_dim = env.observation_space.shape[0]
    num_hidden_layers = 2
    num_neurons_per_layer = 256
    discount = 0.99
    warmup_steps = 1e3
    target_update_freq = 1
    tau = 0.005

    # we ignore frames in between every frame_skip frames since sometimes
    # the game state doesn't change from one frame to the next
    frame_skip = 4

    # observation parameters
    states_per_frame = 4

    state_dim = states_per_frame
    num_actions = env.action_space.n

    replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)

    policy = FC(state_dim, num_actions, num_hidden_layers, num_neurons_per_layer).to(device)
    target_policy = copy.deepcopy(policy)

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_steps = 200000

    state, done, one_round_end = env.reset(), False, False

    state = state[0]
    state = get_state_from_image(state)

    episode_num = 1

    total_reward = 0
    episode_reward = 0

    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    PATH = model_path + 'pong_ddqn.pth'
    PATH_TARGET = model_path + 'pong_ddqn_target.pth'

    for t in range(num_steps):

        # evaluate policy
        if t % 10000 == 0:
            avg_return = eval_policy(env_name, policy, num_actions, frame_skip)
            if avg_return >= 1.0:
                print("Target Pong performance reached, stopping training.")
                break

        if t % 10000 == 0:
            torch.save(policy.state_dict(), PATH)
            torch.save(target_policy.state_dict(), PATH_TARGET)

        if t < warmup_steps:
            action = env.action_space.sample()
        else:
            action = select_action(policy, state, num_actions, eps)



        # Perform action and log results
        # we are using the custom env_step function so we can implement the frame skip functionality
        next_state, reward, done = env_step(env, action, frame_skip)
        next_state = get_state_from_image(next_state)

        if reward != 0:
            one_round_end = True

        episode_reward += reward

        replay_buffer.add(state, action, next_state, reward, done)
        state = copy.copy(next_state)


        if t >= warmup_steps and replay_buffer.size >= batch_size:
            # gradient update
            train_policy(policy, target_policy, replay_buffer, optimizer, discount)

            # update of target network
            if t % target_update_freq == 0:
                with torch.no_grad():
                    for param, target_param in zip(policy.parameters(), target_policy.parameters()):
                        target_param.data.copy_(
                            tau * param.data + (1.0 - tau) * target_param.data
                        )


        # if done, reset environment (plus other accounting)
        if done or one_round_end:
            print(f"Time Steps: {t}, Episode: {episode_num}, Reward: {episode_reward}")
            if done:
                state, done, one_round_end = env.reset(), False, False
                state = state[0]
                state = get_state_from_image(state)

            else:
                one_round_end = False
                state = copy.copy(next_state)

            episode_reward = 0

            episode_num += 1
