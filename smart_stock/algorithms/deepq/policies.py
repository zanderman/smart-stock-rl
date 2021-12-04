from __future__ import annotations
import gym
import numpy as np
import torch
from torch._C import dtype
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

        # Initialize policy network to empty.
        # All sub-classes should override this.
        self.policy_net: torch.nn.Module = None

    def state2tensor(self, state: np.ndarray) -> torch.Tensor:
        """Helper to convert raw observation into PyTorch Tensor with proper dimension for policy network."""
        raise NotImplementedError

    def select_random_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return a randomized action from the action space."""
        # TODO stack actions for multiple states.
        # raise NotImplementedError

        # Convert action to PyTorch Tensor.
        # action = torch.Tensor([action], dtype=action.dtype, device=self.device)

        action = super().select_random_action(obs)
        action = np.array(action).flatten()[0] # Handle random sampling from 1D Box space.

        # Convert action to PyTorch Tensor.
        action = torch.Tensor([action]).to(device=self.device)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:

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
        self.policy_net.to(device) # Send network to desired device.

    def state2tensor(self, state: np.ndarray) -> torch.Tensor:
        """Helper to convert raw observation into PyTorch Tensor with proper dimension for policy network."""
        t = torch.from_numpy(state) # Convert to tensor.
        t = t.to(device=self.device) # Send to device.
        t = t.unsqueeze(0) # Add batch dimension.
        t = t.float() # Convert to float.
        return t

    def select_greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return a greedy action from the action space based on the given observation."""
        print('obs.device',obs.device)
        with torch.no_grad():

            # Get greedy action index.
            action_idx: torch.Tensor = self.policy_net(obs).max(1)[1]

            # Convert action index to action value.
            action:torch.Tensor = torch.from_numpy(self.index2action(action_idx.cpu().numpy())).to(device=self.device)
            # action: torch.Tensor = torch.Tensor([self.index2action(idx) for idx in action_idx], dtype=)

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
        curr_state: torch.Tensor, 
        env: gym.Env, 
        gamma: float,
        alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:

        # # Convert state to tensor.
        # curr_state: torch.Tensor = self.state2tensor(curr_state)
        print('curr_state.device',curr_state.device)

        # Select action according to policy.
        action: torch.Tensor = self.select_action(curr_state)

        # Take selected action and get information from environment.
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.Tensor([reward]).to(device=self.device)

        # Convert state to tensor.
        next_state: torch.Tensor = self.state2tensor(next_state)

        return action, next_state, reward, done