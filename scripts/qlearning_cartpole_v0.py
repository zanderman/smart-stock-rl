"""Q-learning used to solve the CartPole-v0 environment.

Uses linear function approximation in the Fourier basis to map state observations to features vectors.
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
import smart_stock as ss
from typing import List, Tuple


def train(algo: ss.algorithms.qlearning.Q_LFA, env: gym.Env, goal: float, window: int, n_episodes: int = 1000) -> Tuple[List[float], bool]:

    # List of reward values for plotting.
    rewards = []

    # Boolean solution flag.
    found_soln = False

    # Episode loop.
    for i in range(n_episodes):
        reward = algo.run_episode()
        rewards.append(reward)

        # Check if agent has found solution.
        if i >= window and np.mean(rewards[i-window:]) >= goal:
            found_soln = True
            break

    return rewards, found_soln


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def main():
    # Create environment.
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    # Make runs reproduceable.
    RANDOM_SEED = 0 # Turn off by setting as `None`
    if RANDOM_SEED is not None:
        env.seed(RANDOM_SEED)
        env.action_space.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    # Set tweakable parameters.
    gamma = 0.9 # Discount factor (should be in (0,1)).
    alpha = 0.001 # Step size.
    epsilon = 0.005 # Epsilon-greedy action selection (should be in (0,1)).
    n_episodes = 1000 # Upper-limit on number of possible episodes.

    # Initialize linear function approximator by clipping low/high observation range.
    order = 3
    low = np.clip(env.observation_space.low, -10, 10)
    high = np.clip(env.observation_space.high, -10, 10)
    lfa = ss.mapping.fourier.FourierStateFeatureMapping(low, high, order)

    # Create Q-learning algorithm agent with LFA.
    agent = ss.algorithms.qlearning.Q_LFA(env, lfa, gamma, alpha, epsilon)

    # Train the agent 
    goal = 195.0
    window = 100
    rewards, found_soln = train(agent, env, goal, window, n_episodes)

    # Plot moving average.
    plt.figure()
    mr = moving_average(rewards, window)
    if found_soln:
        idx = len(mr) - 1
        mean = mr[-1]
        plt.scatter(x=idx, y=mean, c='r', marker='*', s=50, label=f'solved: episode={idx+window-1}, $\mu$={mean}', zorder=10)
    plt.plot(mr, label='average reward')
    plt.axhline(y=goal, color='g', linestyle='--', label=f'goal is {goal}')
    plt.ylabel('Average Reward')
    plt.xlabel('Sliding Window Index')
    plt.title(f'Moving Average of Reward with Window Size of {window}\nQ-Learning with LFA using Fourier Basis in {env_name} Environment\n$n={order}$, $d={low.size}$, funcs={lfa.coeffs.shape[0]}, $\gamma={gamma}$, $\\alpha={alpha}$, $\epsilon={epsilon}$')
    plt.legend()
    plt.tight_layout()

    # Plot agent performance.
    plt.figure()
    if found_soln:
        idx = len(rewards) - 1
        mean = np.mean(rewards[-window:])
        plt.scatter(x=idx, y=rewards[-1], c='r', marker='*', s=50, label=f'solved: episode={idx}, $\mu$={mean}', zorder=10)
    plt.plot(rewards, label='reward')
    plt.axhline(y=goal, color='g', linestyle='--', label=f'goal is {goal}')
    plt.xlabel('Episode')
    plt.ylabel('Sum of Reward')
    plt.title(f'Sum of Reward per Episode\nQ-Learning with LFA using Fourier Basis in {env_name} Environment\n$n={order}$, $d={low.size}$, funcs={lfa.coeffs.shape[0]}, $\gamma={gamma}$, $\\alpha={alpha}$, $\epsilon={epsilon}$')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()