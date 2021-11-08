import gym
import gym.spaces
import gym.utils.seeding
import numpy as np
from pandas import DataFrame

# CartPole environment example.
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

# Idea: environment for live market interaction.

class StockEnv(gym.Env):
    """Stock Environment.
    
    This environment mimics the behavior of a stock using
    its historical data. Data must be provided in a pandas
    dataframe using proper headers.

    An optional history parameter can be given to control
    the number of days of data to include at once.

    Observation:
        Type: Box(5,<history>)
        Num     Observation     Min     Max
        0       Open            0.       Inf
        1       High            0.       Inf
        2       Low             0.       Inf
        3       Close           0.       Inf
        4       Volume          0.       Inf

    Actions:
        Type: Box(2)
        Num     Action                              Min     Max
        0       Action Selection                    0.      3.
                    < 1 = buy
                    < 2 = sell
                    hold otherwise
        1       Percentage of stock to buy/sell     0.      1.

    Reward:
        pass

    Episode Termination:
        pass

    Data Format:
        The required data format is a table of the form:

            Open,High,Low,Close,Volume
            24.333,24.333,23.946,23.946,43321

        The first row must be a header with the same names as above.
        Every subsequent row must contain the desired data.
    """

    def __init__(self, 
        df: DataFrame,
        start_balance: float,
        history: int = 5, 
    ):
        super.__init__()

        # Initial reset parameters.
        self._history = history
        self._start_balance = start_balance

        self.df = df
        self.action_space = gym.spaces.Box(
            low=np.array([0., 0.]),
            high=np.array([3., 1.]),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=0.,
            high=np.finfo(np.float32).max,
            shape=(5,history,),
            dtype=np.float32,
        )

        # Seed the environment.
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        pass

    def reset(self):
        self.balance = self._start_balance

    def render(self, mode: str = 'human'):
        pass