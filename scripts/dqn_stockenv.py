
from __future__ import annotations
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import smart_stock as ss
import torch

from smart_stock.algorithms.deepq import policies


# df = dataset['aapl']
# start_balance = 100
# history = 5
# env = ss.envs.StockEnv(df, start_balance, history)


def train(
    agent: ss.algorithms.deepq.dqn.DQN, 
    env: gym.Env,
    max_episodes: int = 1000,
    max_steps: int = None,
    batch_size: int = 32,
    memory_capacity: int = 1000,
    target_update_freq: int = 10,
    render: bool = False,
    render_mode: str = None,
    ) -> tuple[list[float], bool]:

    # List of reward values for plotting.
    rewards = []

    # Boolean solution flag.
    found_soln = False

    # Episode loop.
    for i in range(max_episodes):
        reward = agent.run_episode(
            max_steps=max_steps, 
            target_update_freq=target_update_freq,
            render=render, 
            render_mode=render_mode,
        )
        rewards.append(reward)
        # if i%100 == 0: print(f'[{i}] {reward}')
        print(f'[{i}] {reward}')

    return rewards, found_soln

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def main():

    # Prepare dataset.
    path = os.path.expanduser('~/Desktop')
    dataset = ss.datasets.HugeStockMarketDataset(path)

    # Create stock environment using specific stock.
    df = dataset['aapl']
    start_balance = 100
    max_stock = 1 # 100
    start_day = None
    env_name = 'StockDataEnv'
    env = ss.envs.StockDataEnv(
        df=df, 
        start_balance=start_balance, 
        max_stock=max_stock, 
        start_day=start_day,
    )

    # Make runs reproduceable.
    RANDOM_SEED = 0 # Turn off by setting as `None`
    if RANDOM_SEED is not None:
        env.seed(RANDOM_SEED)
        env.action_space.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    # Set tweakable parameters.
    gamma = 0.9 # Discount factor (should be in (0,1)).
    alpha = 0.0001 # Step size.
    epsilon = 0.2 # Epsilon-greedy action selection (should be in (0,1)).
    max_episodes = 50 # 1000 # Upper-limit on number of possible episodes.
    max_steps = 500
    batch_size = 32
    memory_capacity = 1000
    target_update_freq = 10
    render = False
    render_mode = 'csv'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_actions = len(np.arange(env.action_space.low, env.action_space.high+1))
    dims = [
        env.observation_space.shape[0],
        512,
        n_actions
    ]
    policy = ss.algorithms.deepq.policies.FeedForwardLinearPolicy(
        env.action_space,
        env.observation_space,
        epsilon,
        device,
        dims,
        )

    optimizer = torch.optim.SGD(policy.policy_net.parameters(), lr=alpha)
    criterion = torch.nn.SmoothL1Loss()

    agent = ss.algorithms.deepq.dqn.DQN(
        env, 
        policy, 
        gamma, 
        alpha, 
        memory_capacity, 
        batch_size,
        optimizer,
        criterion
    )

    # max_steps = 1000
    # agent.run_episode(max_steps)

    # Train the agent 
    rewards, found_soln = train(
        agent, 
        env, 
        max_episodes, 
        max_steps, 
        batch_size,
        memory_capacity,
        target_update_freq,
        render, 
        render_mode,
    )



    # return

    # # ---------------------------------

    # # Initialize linear function approximator by clipping low/high observation range.
    # order = 3
    # # print(env.observation_space.low.shape)
    # # print(env.observation_space.high.shape)
    # obs_low = np.clip(env.observation_space.low, -10, 10)
    # # low = np.clip(env.observation_space.low[0], -10, 10)
    # obs_high = np.clip(env.observation_space.high, -10, 10)
    # # high = np.clip(env.observation_space.high[0], -10, 10)
    # # lfa = ss.mapping.fourier.UncoupledFourierStateFeatureMapping(low, high, order)
    # lfa = ss.mapping.fourier.FourierStateFeatureMapping(obs_low, obs_high, order)

    # # # Get range of action space.
    # # act_low = env.action_space.low
    # # act_high = env.action_space.high

    # # obs = env.reset()
    # # # print(obs.shape)
    # # # print(lfa.normalize(obs).shape)
    # # # print(lfa(obs).shape)

    # # print('obs.shape',obs.shape)
    # # print('lfa.normalize(obs).shape',lfa.normalize(obs).shape)
    # # print('lfa.coeffs.shape',lfa.coeffs.shape)
    # # print('lfa.coeffs',lfa.coeffs)
    # # print(lfa(obs).shape)

    # # Create Q-learning algorithm agent with LFA.
    # agent = ss.algorithms.qlearning.Q_SFM(env, lfa, gamma, alpha, epsilon)

    # # Train the agent 
    # rewards, found_soln = train(agent, env, max_episodes, max_steps, render, render_mode)

    # Plot the rewards.
    plt.figure()
    plt.plot(rewards)
    plt.title(f"Sum of Reward per Episode\nDQN using FeedForwardLinear Network in {env_name} Environment\n$\gamma={gamma}$, $\\alpha={alpha}$, $\epsilon={epsilon}$")
    plt.xlabel('Episode')
    plt.ylabel('Sum of Reward')
    plt.tight_layout()
    plt.show()

    # # Plot moving average.
    # plt.figure()
    # mr = moving_average(rewards, window)
    # if found_soln:
    #     idx = len(mr) - 1
    #     mean = mr[-1]
    #     plt.scatter(x=idx, y=mean, c='r', marker='*', s=50, label=f'solved: episode={idx+window-1}, $\mu$={mean}', zorder=10)
    # plt.plot(mr, label='average reward')
    # plt.axhline(y=goal, color='g', linestyle='--', label=f'goal is {goal}')
    # plt.ylabel('Average Reward')
    # plt.xlabel('Sliding Window Index')
    # plt.title(f'Moving Average of Reward with Window Size of {window}\nQ-Learning with LFA using Fourier Basis in {env_name} Environment\n$n={order}$, $d={low.size}$, funcs={lfa.coeffs.shape[0]}, $\gamma={gamma}$, $\\alpha={alpha}$, $\epsilon={epsilon}$')
    # plt.legend()
    # plt.tight_layout()

    # # Plot agent performance.
    # plt.figure()
    # if found_soln:
    #     idx = len(rewards) - 1
    #     mean = np.mean(rewards[-window:])
    #     plt.scatter(x=idx, y=rewards[-1], c='r', marker='*', s=50, label=f'solved: episode={idx}, $\mu$={mean}', zorder=10)
    # plt.plot(rewards, label='reward')
    # plt.axhline(y=goal, color='g', linestyle='--', label=f'goal is {goal}')
    # plt.xlabel('Episode')
    # plt.ylabel('Sum of Reward')
    # plt.title(f'Sum of Reward per Episode\nQ-Learning with LFA using Fourier Basis in {env_name} Environment\n$n={order}$, $d={low.size}$, funcs={lfa.coeffs.shape[0]}, $\gamma={gamma}$, $\\alpha={alpha}$, $\epsilon={epsilon}$')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    main()