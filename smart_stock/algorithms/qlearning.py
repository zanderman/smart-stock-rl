import gym
import numpy as np
from typing import Tuple
from ..mapping.state_feature_mapping import StateFeatureMapping



class Q_SFM:
    """Q-learning with linear function approximation using the Fourier basis."""

    def __init__(self, env: gym.Env, sfm: StateFeatureMapping, gamma: float, alpha: float, epsilon: float):
        self.env = env
        self.sfm = sfm
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # Initialize weight matrix.
        self.theta = np.zeros((env.action_space.n, sfm.n_features,)) # actions x features


    def q_value(self, obs: np.ndarray, action: int = None) -> np.ndarray:
        """Obtain current Q-value for given observation and action.

        If no action is specified, it returns the Q-value for all actions.
        """
        phi = self.sfm(obs) # Compute LFA for observation.

        # Return matrix of Q-values for all actions.
        if action is None:
            return np.array([np.dot(phi, self.theta[a]) for a in range(self.env.action_space.n)])

        # Return Q-value for given action.
        else:
            return np.dot(phi, self.theta[action])


    def step(self, curr_state: np.ndarray) -> Tuple[np.ndarray, float, bool]:

        # Epsilon-greedy action selection.
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax([self.q_value(curr_state, a) for a in range(self.env.action_space.n)])

        # Take selected action and get information from environment.
        next_state, reward, done, _ = self.env.step(action)

        # Weight update.
        delta = reward + self.gamma * np.max(self.q_value(next_state), axis=0) - self.q_value(curr_state, action)
        self.theta[action] += self.alpha * delta * self.sfm(curr_state)

        return next_state, reward, done


    def run_episode(self):

        # Reset the environment and get starting state.
        curr_state = self.env.reset()

        total_reward = 0
        while True:

            # Step the algorithm through the current state and retreive
            # the Q-matrix, next state, and the termination flag.
            next_state, reward, done = self.step(curr_state)

            # Accumulate rewards for the current episode.
            total_reward += reward

            # Update the current state.
            curr_state = next_state

            # Terminate steps early if environment enters terminal state.
            if done: break

        return total_reward