from __future__ import annotations
import gym
import numpy as np
from ..basepolicy import ContinuousStateDiscreteActionPolicy
from ...mapping.fourier import FourierStateFeatureMapping


class QPolicy(ContinuousStateDiscreteActionPolicy):
    """Generic Q-learning policy.

    This policy accepts the following:

        Action Space:
            - Discrete
            - Continuous integers

        State space:
            - Discrete
            - Continuous
    """
    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
        n_features: int,
        epsilon: float,
    ):
        super().__init__(action_space, observation_space)

        # Preserve Q-learing-specific members.
        self.epsilon = epsilon

        # Initialize weight matrix.
        self.theta = np.zeros((self.action_count, n_features,)) # actions x features

    def features(self, obs: np.ndarray) -> np.ndarray:
        """Retreive feature representation of observation."""
        return obs

    def q_value(self, obs: np.ndarray, action: int = None) -> np.ndarray:
        """Obtain current Q-value for given observation and action.

        If no action is specified, it returns the Q-value for all actions.
        """
        phi = self.features(obs) # Compute LFA for observation.

        # Return matrix of Q-values for all actions.
        if action is None:
            return np.array([np.dot(phi, self.theta[ai]) for ai in range(self.action_count)])

        # Return Q-value for given action.
        else:
            return np.dot(phi, self.theta[self.action2index(action)])

    def step(self, 
        curr_state: np.ndarray, 
        env: gym.Env, 
        gamma: float,
        alpha: float
    ) -> tuple[np.ndarray, float, bool]:

        # Epsilon-greedy action selection.
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()
            action = np.array(action).flatten()[0] # Handle random sampling from 1D Box space.
        else:
            # q = [(self.index2action(ai), ai, self.q_value(curr_state, self.index2action(ai))) for ai in range(self.action_count)]
            # print('Q:',q)

            # Compute Q values.
            q = [self.q_value(curr_state, self.index2action(ai)) for ai in range(self.action_count)]

            # Get index of all maximum Q values.
            max_q_idxs = np.where(np.isclose(q, np.max(q)))[0]

            # Randomly select a maximum to prevent always selecting first instance.
            action_index = np.random.choice(max_q_idxs)

            # OLD IMPLEMENTATION OF SELECTING FIRST MAX.
            # action_index = np.argmax([self.q_value(curr_state, self.index2action(ai)) for ai in range(self.action_count)])

            action = self.index2action(action_index) # Convert index to action value.

        # Take selected action and get information from environment.
        next_state, reward, done, _ = env.step(action)

        # Weight update.
        delta = reward + gamma * np.max(self.q_value(next_state), axis=0) - self.q_value(curr_state, action)
        self.theta[action] += alpha * delta * self.features(curr_state)

        return next_state, reward, done



class FourierPolicy(QPolicy):
    """Q-learning policy using linear function approximation with Fourier basis for state feature estimation."""
    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
        order: int,
        epsilon: float,
        low: np.ndarray = None,
        high: np.ndarray = None,
    ):

        # Build state-feature mapping class.
        if low is None:
            low = np.clip(observation_space.low, -10, 10)
        if high is None:
            high = np.clip(observation_space.high, -10, 10)
        self.lfa = FourierStateFeatureMapping(low, high, order)
        self.order = order

        # Instantiate parent class.
        super().__init__(action_space, observation_space, self.lfa.n_features, epsilon)

    def features(self, obs: np.ndarray) -> np.ndarray:
        """Compute LFA for observation."""
        return self.lfa(obs) # Compute LFA for observation.