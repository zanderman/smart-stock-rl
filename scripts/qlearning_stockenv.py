
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import smart_stock as ss
from typing import List, Tuple


# df = dataset['aapl']
# start_balance = 100
# history = 5
# env = ss.envs.StockEnv(df, start_balance, history)


def train(
    algo: ss.algorithms.qlearning.Q_SFM, 
    env: gym.Env,
    max_episodes: int = 1000,
    max_steps: int = None,
    render: bool = False
    ) -> Tuple[List[float], bool]:

    # List of reward values for plotting.
    rewards = []

    # Boolean solution flag.
    found_soln = False

    # Episode loop.
    for i in range(max_episodes):
        reward = algo.run_episode(max_steps=max_steps, render=render)
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
    max_stock = 100
    start_day = None
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
    alpha = 0.01 # Step size.
    epsilon = 0.2 # Epsilon-greedy action selection (should be in (0,1)).
    max_episodes = 10 # 1000 # Upper-limit on number of possible episodes.
    max_steps = 100
    render = False

    # Initialize linear function approximator by clipping low/high observation range.
    order = 3
    # print(env.observation_space.low.shape)
    # print(env.observation_space.high.shape)
    obs_low = np.clip(env.observation_space.low, -10, 10)
    # low = np.clip(env.observation_space.low[0], -10, 10)
    obs_high = np.clip(env.observation_space.high, -10, 10)
    # high = np.clip(env.observation_space.high[0], -10, 10)
    # lfa = ss.mapping.fourier.UncoupledFourierStateFeatureMapping(low, high, order)
    lfa = ss.mapping.fourier.FourierStateFeatureMapping(obs_low, obs_high, order)

    # # Get range of action space.
    # act_low = env.action_space.low
    # act_high = env.action_space.high

    # obs = env.reset()
    # # print(obs.shape)
    # # print(lfa.normalize(obs).shape)
    # # print(lfa(obs).shape)

    # print('obs.shape',obs.shape)
    # print('lfa.normalize(obs).shape',lfa.normalize(obs).shape)
    # print('lfa.coeffs.shape',lfa.coeffs.shape)
    # print('lfa.coeffs',lfa.coeffs)
    # print(lfa(obs).shape)

    # Create Q-learning algorithm agent with LFA.
    agent = ss.algorithms.qlearning.Q_SFM(env, lfa, gamma, alpha, epsilon)

    # Train the agent 
    rewards, found_soln = train(agent, env, max_episodes, max_steps, render)

    # Plot the rewards.
    plt.figure()
    plt.plot(rewards)
    plt.legend()
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