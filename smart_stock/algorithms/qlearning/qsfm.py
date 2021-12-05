from __future__ import annotations
import gym
import numpy as np
from ...mapping.state_feature_mapping import StateFeatureMapping



class QSFM:
    """Q-learning with linear function approximation using the Fourier basis."""

    def __init__(self, 
        env: gym.Env,  
        sfm: StateFeatureMapping, 
        gamma: float, 
        alpha: float, 
        epsilon: float,
    ):
        self.sfm = sfm
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # Generate discrete action list based on integer Box space.
        if type(env.action_space) == gym.spaces.Box:

            # Validate action space data type and limits.
            if env.action_space.dtype != int:
                raise TypeError('only integer action space is supported')
            elif env.action_space.high >= np.iinfo(env.action_space.dtype).max:
                raise TypeError('action space upper limit must be finite')
            elif env.action_space.low <= np.iinfo(env.action_space.dtype).min:
                raise TypeError('action space lower limit must be finite')

            # Get action space limits.
            actions_low = env.action_space.low
            actions_high = env.action_space.high

        # Generate discrete action list based on Discrete space.
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


    def optimize_policy(self,
        curr_state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        ):

        # Weight update.
        delta = reward + self.gamma * np.max(self.q_value(next_state), axis=0) - self.q_value(curr_state, action)

        # Clip the error to prevent outliers from exploding.
        delta = np.clip(delta, -1, 1)

        # Update weights.
        self.theta[action] += self.alpha * delta * self.sfm(curr_state)


    def step(self, 
        env: gym.Env,
        curr_state: np.ndarray
        ) -> tuple[np.ndarray, int, np.ndarray, float, bool]:

        # Epsilon-greedy action selection.
        if np.random.uniform(0, 1) < self.epsilon:
            action = env.action_space.sample()
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

        # # Weight update.
        # delta = reward + self.gamma * np.max(self.q_value(next_state), axis=0) - self.q_value(curr_state, action)

        # # Clip the error to prevent outliers from exploding.
        # delta = np.clip(delta, -1, 1)

        # # Update weights.
        # self.theta[action] += self.alpha * delta * self.sfm(curr_state)

        return curr_state, action, next_state, reward, done


    def run_episode(self,
        env: gym.Env,  
        max_steps: int = None, 
        render: bool = False, 
        render_mode: str = None, 
        optimize: bool = True,
        ):

        # Reset the environment and get starting state.
        curr_state = env.reset()

        total_reward = 0
        step = 0
        while True:

            # Render the environment if requested.
            if render: env.render(mode=render_mode)

            # Step the algorithm through the current state and retreive
            # the Q-matrix, next state, and the termination flag.
            curr_state, action, next_state, reward, done = self.step(env, curr_state)

            # Optimize the policy at the current step.
            if optimize:
                self.optimize_policy(curr_state, action, next_state, reward)

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