"""Deep Q-Learning agents.
"""
from collections import deque, namedtuple
from typing import Callable
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