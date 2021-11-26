"""Deep Q-Learning agents.
"""
from __future__ import annotations
from collections import deque, namedtuple
from typing import List, Tuple
import torch
import random


# We use a `namedtuple` object here because it can
# easily be used inside a PyTorch `Tensor` for vectorization.
StateActionTransition = namedtuple(
    'StateActionTransition',
    [
        'state',
        'action',
        'next_state',
        'reward',
    ],
)


class ReplayMemory(list):
    """State-action transition replay memory.

    This class stores a collection of state-action transitions
    for use with RL agents when traversing environments.
    """
    def __init__(self, capacity: int):
        super().__init__()
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index: int):
        return list(self.memory)[index]

    def __setitem__(self, index: int, newvalue: object):
        self.memory[index] = newvalue

    def __delitem__(self, index: int):
        del self.memory[index]

    def __iter__(self):
        for item in self.memory:
            yield item

    def __str__(self):
        return str(list(self.memory))

    def push(self, *args, **kwargs):
        """Preserve a state-action transition in memory."""
        self.memory.append(StateActionTransition(*args, **kwargs))

    def pop(self):
        """Retrieve last state-action transition from memory."""
        return self.memory.pop()

    def sample(self, size: int):
        """Generate a random sampling without replacement from replay memory."""
        return random.sample(self.memory, size)


# Example: https://gist.github.com/kkweon/52ea1e118101eb574b2a83b933851379
# Example: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class DQN_Linear(torch.nn.Module):
    """Deep Q-learning linear feed-forward network."""
    def __init__(self, dims: list[int]):
        super().__init__()

        # Preserve dimension list.
        self.dims = dims

        # Define list of layers.
        self.layers = torch.nn.ModuleList()

        # Build network using dimension list.
        # The input/output dimensions are collected by
        # zipping the original list with a shift-by-1 version.
        for input_dim, output_dim in zip(dims, dims[1:]):
            self.layers.append(torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim),
                torch.nn.BatchNorm1d(output_dim),
                torch.nn.PReLU(), # https://arxiv.org/pdf/1710.05941.pdf
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Sequentially call forward functions of each intermediate layer.
        # This cascades the input through the entire network.
        for layer in self.layers:
            x = layer(x)

        # Return cascaded result.
        return x
