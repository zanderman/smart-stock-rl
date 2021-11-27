from __future__ import annotations
import gym
import numpy as np
import torch
from ..basepolicy import ContinuousStateDiscreteActionPolicy


class DQNPolicy(ContinuousStateDiscreteActionPolicy):
    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
        epsilon: float,
        device: torch.device,
    ):
        super().__init__(action_space, observation_space)

        # Preserve DQN-specific members.
        self.epsilon = epsilon
        self.device = device

    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute probabilities for each action using the given observation."""
        pass

    def step(self, 
        curr_state: np.ndarray, 
        env: gym.Env, 
        gamma: float,
        alpha: float
    ) -> tuple[np.ndarray, float, bool]:

        # Random action.
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()
            action = np.array(action).flatten()[0] # Handle random sampling from 1D Box space.

            # Convert action to PyTorch Tensor.
            action = torch.Tensor([action], dtype=action.dtype, device=self.device)

        # Epsilon-greedy action selection.
        else:
            # TODO get action from network.
            pass

        # Take selected action and get information from environment.
        next_state, reward, done, _ = env.step(action)

        return next_state, reward, done