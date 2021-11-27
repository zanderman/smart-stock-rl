import gym


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