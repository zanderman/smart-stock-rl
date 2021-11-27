"""Deep Q-Learning agents.
"""
from __future__ import annotations
from collections import deque, namedtuple
from typing import Callable
import gym
import random
import torch
from .policies import DQNPolicy


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


class DQN:
    def __init__(self, 
        env: gym.Env,  
        policy: DQNPolicy, 
        gamma: float, 
        alpha: float, 
        memory_capacity: int,
        batch_size: int,
    ):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.alpha = alpha
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size

        # Keep another policy as the target policy for stability.
        self.target_policy_network: torch.nn.Module = self.policy.policy_net.detach().clone()
        self.target_policy_network.load_state_dict(self.policy.policy_net.state_dict())
        self.target_policy_network.eval()

    def optimize_policy(self):
        """Optimize the DQN policy using replay memory."""

        # Do not update if there is not enough transitions in memory.
        if len(self.memory) < self.batch_size:
            return

        # Randomly select transitions from replay memory.
        batch = self.memory.sample(self.batch_size)

        # Transpose batch to get separate lists.
        batch = StateActionTransition(*zip(*batch))

        # Convert batch lists to PyTorch tensors.
        batch_states = torch.cat(batch.state)
        batch_actions = torch.cat(batch.action)
        batch_rewards = torch.cat(batch.reward)
        batch_next_states = torch.cat(batch.next_state)

        print('optimize_policy')
        print('batch_states.shape',batch_states.shape)
        print('batch_actions.shape',batch_actions.shape)
        print('batch_rewards.shape',batch_rewards.shape)
        print('batch_next_states.shape',batch_next_states.shape)
        print()

    def run_episode(self, 
        max_steps: int = None, 
        render: bool = False, 
        render_mode: str = None,
    ):

        # Reset the environment and get starting state.
        curr_state = self.env.reset()

        # Convert state to tensor.
        curr_state: torch.Tensor = self.policy.state2tensor(curr_state)

        total_reward = 0
        step = 0
        while True:

            # Render the environment if requested.
            if render: self.env.render(mode=render_mode)

            # Step the algorithm through the current state and retreive
            # the Q-matrix, next state, and the termination flag.
            action, next_state, reward, done = self.policy.step(
                curr_state, 
                self.env, 
                self.gamma, 
                self.alpha,
            )

            # Store state-action transition in memory.
            self.memory.push(
                curr_state,
                action,
                next_state,
                reward,
            )

            # Optimize the policy at the current step.
            self.optimize_policy()

            # Accumulate rewards for the current episode.
            total_reward += reward.item()

            # Update the current state.
            curr_state = next_state

            # Update the step count.
            step += 1

            # Terminate steps early if environment enters terminal state.
            if done: break

            # Terminate if step count is reached.
            elif max_steps is not None and step >= max_steps: break

        return total_reward