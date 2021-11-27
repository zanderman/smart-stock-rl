from __future__ import annotations
import gym
import numpy as np
import torch
from ..basepolicy import ContinuousStateDiscreteActionPolicy, EpsilonGreedyPolicy
from .networks import FeedForwardLinear


class DQNPolicy(EpsilonGreedyPolicy, ContinuousStateDiscreteActionPolicy):
    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
        epsilon: float,
        device: torch.device,
    ):
        super().__init__(action_space, observation_space, epsilon)

        # Preserve DQN-specific members.
        self.device = device

    def select_random_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return a randomized action from the action space."""
        # TODO stack actions for multiple states.
        # raise NotImplementedError

        # Convert action to PyTorch Tensor.
        # action = torch.Tensor([action], dtype=action.dtype, device=self.device)

        action = super().select_random_action(obs)
        action = np.array(action).flatten()[0] # Handle random sampling from 1D Box space.

        # Convert action to PyTorch Tensor.
        action = torch.Tensor([[action]], dtype=action.dtype, device=self.device)

        return action

    def select_greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return a greedy action from the action space based on the given observation."""
        raise NotImplementedError

    def q_value(self, obs: torch.Tensor, action: int = None) -> torch.Tensor:
        """Compute Q-value for given observation and action.

        If no action is specified, it returns the Q-values for all actions.
        """
        raise NotImplementedError

    def step(self, 
        curr_state: np.ndarray, 
        env: gym.Env, 
        gamma: float,
        alpha: float
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:

        # Select action according to policy.
        action = self.select_action(curr_state)

        # Take selected action and get information from environment.
        next_state, reward, done, _ = env.step(action)

        # TODO Update policy.
        pass

        return action, next_state, reward, done


class FeedForwardLinearPolicy(DQNPolicy):
    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
        epsilon: float,
        device: torch.device,
        dims: list[int],
    ):
        super().__init__(action_space, observation_space, epsilon, device)

        # Create network.
        self.policy_net = FeedForwardLinear(dims)

    @staticmethod
    def state2tensor(state: np.ndarray) -> torch.Tensor:
        """Helper to convert raw observation into PyTorch Tensor."""
        t = torch.from_numpy(state) # Convert to tensor.
        t = t.unsqueeze(0) # Add batch dimension.
        t = t.float() # Convert to float.
        return t

    def select_greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return a greedy action from the action space based on the given observation."""
        print('greedy action')
        with torch.no_grad():
            print('obs.shape',obs.shape,obs.dtype)
            action = self.policy_net(obs)
            print(action.shape)
            return action

    def q_value(self, obs: torch.Tensor, action: int = None) -> torch.Tensor:
        """Compute Q-value for given observation and action.

        If no action is specified, it returns the Q-values for all actions.
        """
        q_values = self.policy_net(obs)
        if action is None:
            return q_values
        else:
            return q_values[action]

    def step(self, 
        curr_state: np.ndarray, 
        env: gym.Env, 
        gamma: float,
        alpha: float
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:

        # Convert state to tensor.
        curr_state = self.state2tensor(curr_state)
        print('curr_state.shape',curr_state.shape,curr_state.dtype)

        # Select action according to policy.
        action = self.select_action(curr_state)

        # Take selected action and get information from environment.
        next_state, reward, done, _ = env.step(action)

        return action, next_state, reward, done