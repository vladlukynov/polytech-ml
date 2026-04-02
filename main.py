from collections import namedtuple, deque

from itertools import count

from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

import gymnasium as gym
import gymnasium as gym

import numpy as np

import os

import random

import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


GAMMA = 0.99
LR = 1e-4
TAU = 0.005

EPSILON = 1.0  # Start with full exploration
EPSILON_MIN = 0.01  # Minimum value
EPSILON_DECAY = 0.995  # Decay factor per episode

TRANSITION = namedtuple(
    "Transition", ["state", "action", "next_state", "reward", "done"]
)

MODEL_FILE_NAME = 'model.bin'


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(TRANSITION(*args))

    def sample(self, batch_size):
        return (
            random.sample(self.memory, batch_size)
            if batch_size < len(self.memory)
            else self.memory
        )

    def __len__(self):
        return len(self.memory)


def select_action(state, env, device, policy_net):
    if np.random.rand() < EPSILON:
        return torch.tensor(
            [[env.action_space.sample()]], dtype=torch.long, device=device
        )
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)  # Exploit (best action)


def initialize_learn_env():
    global EPSILON

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    num_episodes = 600
    batch_size = 128

    reward_list = []
    episode_durations = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Инициалзиция окружения
    env = gym.make("LunarLander-v3", render_mode='human')

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    replay_memory = ReplayMemory(10000)

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss()

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0

        for t in count():
            action = select_action(state=state, env=env, device=device, policy_net=policy_net)
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            done = terminated or truncated
            reward = torch.tensor([reward], device=device)
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            replay_memory.push(state, action, next_state, reward, done)

            state = next_state
            total_reward += reward.item()

            if len(replay_memory) >= batch_size:
                transitions = replay_memory.sample(batch_size)
                states, actions, next_states, rewards, dones = zip(*transitions)

                states_batch = torch.cat(states)
                next_states_batch = torch.cat(next_states)
                actions_batch = torch.cat(actions)
                rewards = torch.tensor(rewards, device=device)
                dones = torch.tensor(dones, device=device)

                q_target = (
                    GAMMA * target_net(next_states_batch).detach().max(1)[0] * ~dones
                    + rewards
                )
                q_policy = policy_net(states_batch).gather(1, actions_batch)

                # Calculate the Huber loss
                loss = criterion(q_policy, q_target.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()

                # In-place gradient clipping to stabilize training
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

                optimizer.step()

                # Log loss to TensorBoard
                writer.add_scalar("Loss", loss.item(), episode)

            # Update target network
            for target_param, main_param in zip(
                target_net.parameters(), policy_net.parameters()
            ):
                target_param.data.copy_(
                    TAU * main_param.data + (1 - TAU) * target_param.data
                )

            if done:
                episode_durations.append(t + 1)
                reward_list.append(total_reward)

                # Log metrics to TensorBoard
                writer.add_scalar("Reward", total_reward, episode)
                writer.add_scalar("Episode Duration", t + 1, episode)
                writer.add_scalar("Epsilon", EPSILON, episode)
                break

        # Decay epsilon
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    # Close TensorBoard writer
    writer.close()

    # Save the trained model
    torch.save(policy_net.state_dict(), MODEL_FILE_NAME)
    print("Model saved successfully!")

    print("Complete")
    env.close()

def initialize_demo_env():
    num_episodes = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize the environment
    env = gym.make("LunarLander-v3", render_mode="human")

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize the model architecture
    policy_net = DQN(n_observations, n_actions).to(device)

    # Load the trained weights
    policy_net.load_state_dict(torch.load(MODEL_FILE_NAME))
    policy_net.eval()  # Set the model to evaluation mode

    print("Model loaded successfully!")

    for _ in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        for t in count():
            with torch.no_grad():
                action = (
                    policy_net(state).max(1).indices.view(1, 1)
                )  # Exploit (best action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            total_reward += reward
            done = terminated or truncated
            if done:
                print(total_reward)
                break


def main():
    if os.path.isfile(MODEL_FILE_NAME):
        print("Model founded, start demo!")
        initialize_demo_env()
    else:
        print("Model not founded, start learning!")
        initialize_learn_env()


if __name__ == '__main__':
    main()
