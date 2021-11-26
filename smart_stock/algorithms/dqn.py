"""Deep Q-Learning agents.
"""
from collections import deque, namedtuple
import torch


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


