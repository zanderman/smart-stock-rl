import gym
import numpy as np
from typing import Tuple
from ..mapping.state_feature_mapping import StateFeatureMapping



class Q_SFM:
    """Q-learning with linear function approximation using the Fourier basis."""

    def __init__(self, 
        env: gym.Env,  
        sfm: StateFeatureMapping, 
        gamma: float, 
        alpha: float, 
        epsilon: float,
    ):
        self.env = env
        self.sfm = sfm
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # Generate discrete action list based on action space type.
        if type(env.action_space) == gym.spaces.Box:
            actions_low = env.action_space.low
            actions_high = env.action_space.high
        if type(env.action_space) == gym.spaces.Discrete:
            actions_low = 0
            actions_high = env.action_space.n

        # Build action list.
        self.action_list = np.arange(actions_low, actions_high+1)
        self.action_count = len(self.action_list)

        # Initialize weight matrix.
        self.theta = np.zeros((self.action_count, sfm.n_features,)) # actions x features


    def index2action(self, index: int):
        """Retreive action from position index."""
        return self.action_list[index]


    def action2index(self, action: int):
        """Retreive position index from action."""
        return np.where(self.action_list == action)[0][0]


    def q_value(self, obs: np.ndarray, action: int = None) -> np.ndarray:
        """Obtain current Q-value for given observation and action.

        If no action is specified, it returns the Q-value for all actions.
        """
        phi = self.sfm(obs) # Compute LFA for observation.

        # Return matrix of Q-values for all actions.
        if action is None:
            return np.array([np.dot(phi, self.theta[ai]) for ai in range(self.action_count)])

        # Return Q-value for given action.
        else:
            return np.dot(phi, self.theta[self.action2index(action)])


    def step(self, curr_state: np.ndarray) -> Tuple[np.ndarray, float, bool]:

        # Epsilon-greedy action selection.
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action_index = np.argmax([self.q_value(curr_state, self.index2action(ai)) for ai in range(self.action_count)])
            action = self.index2action(action_index) # Convert index to action value.

        # Take selected action and get information from environment.
        next_state, reward, done, _ = self.env.step(action)

        # Weight update.
        delta = reward + self.gamma * np.max(self.q_value(next_state), axis=0) - self.q_value(curr_state, action)
        self.theta[action] += self.alpha * delta * self.sfm(curr_state)

        return next_state, reward, done


    def run_episode(self, max_steps: int = None, render: bool = False):

        # Reset the environment and get starting state.
        curr_state = self.env.reset()

        total_reward = 0
        step = 0
        while True:

            # Render the environment if requested.
            if render: self.env.render()

            # Step the algorithm through the current state and retreive
            # the Q-matrix, next state, and the termination flag.
            next_state, reward, done = self.step(curr_state)

            # Accumulate rewards for the current episode.
            total_reward += reward

            # Update the current state.
            curr_state = next_state

            # Update the step count.
            step += 1

            # Terminate steps early if environment enters terminal state.
            if done: break

            # Terminate if step count is reached.
            elif max_steps is not None and step >= max_steps: break

        return total_reward