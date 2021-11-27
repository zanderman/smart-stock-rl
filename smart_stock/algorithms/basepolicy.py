import gym
import numpy as np

class BasePolicy:
    """Abstract base class for policies.
    """
    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
    ):
        self.action_space = action_space
        self.observation_space = observation_space

    def step(self, 
        curr_state: object,
        env: gym.Env,
        *args,
        **kwargs,
    ):
        raise NotImplementedError


class ContinuousStateDiscreteActionPolicy(BasePolicy):
    """Base policy for continuous state space and discrete action space.

    This policy also supports action spaces that are continuous boxes
    with integer values; thereby being finite and discrete. These
    spaces are interpreted as discrete spaces with a finite number
    of actions.
    """
    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
    ):
        super().__init__(action_space, observation_space)

        # Generate discrete action list based on integer Box space.
        if type(action_space) == gym.spaces.Box:

            # Validate action space data type and limits.
            if action_space.dtype != int:
                raise TypeError('only integer action space is supported')
            elif action_space.high >= np.iinfo(action_space.dtype).max:
                raise TypeError('action space upper limit must be finite')
            elif action_space.low <= np.iinfo(action_space.dtype).min:
                raise TypeError('action space lower limit must be finite')

            # Get action space limits.
            actions_low = action_space.low
            actions_high = action_space.high

        # Generate discrete action list based on Discrete space.
        if type(action_space) == gym.spaces.Discrete:
            actions_low = 0
            actions_high = action_space.n

        # Build action list.
        self.action_list = np.arange(actions_low, actions_high+1)
        self.action_count = len(self.action_list)

    def index2action(self, index: int):
        """Retreive action from position index."""
        return self.action_list[index]

    def action2index(self, action: int):
        """Retreive position index from action."""
        return np.where(self.action_list == action)[0][0]