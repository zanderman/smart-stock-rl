import enum
import gym
import gym.spaces
import gym.utils.seeding
import numpy as np
from pandas import DataFrame

# CartPole environment example.
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

# Idea: environment for live market interaction.

class ActionType(enum.Enum):
    BUY = 0
    SELL = 1
    HOLD = 2


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
        self._data_window = self._history + 1
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


    def _action_buy(self, 
        stock_price: float, 
        action_percent_amount: float,
        ):

        # Total number of shares the agent can purchase.
        total_possible_shares = int(self.balance / stock_price)

        # Amount of shares the agent purchases given
        # their percentage amount.
        bought_shares = int(total_possible_shares * action_percent_amount)

        # Compute total purchase cost and cost basis for tax purposes.
        cost = bought_shares * stock_price
        self.cost_basis = (cost + self.cost_basis*self.shares)/(self.shares + bought_shares)

        # Update number of shares held.
        self.shares += bought_shares


    def _perform_action(self, action):

        # Set current stock price at a random value 
        # between the start and end price for the day.
        stock_price = self.np_random.uniform(
            low=self.df['Open'].iloc[self.current_step],
            high=self.df['Close'].iloc[self.current_step],
        )

        # Split the action into type and percentage amount.
        action_type = ActionType(int(action[0]))
        action_percent_amount = action[1]

        if action_type == ActionType.BUY:
            self._action_buy(stock_price, action_percent_amount)

        elif action_type == ActionType.SELL:
            pass

        elif action_type == ActionType.HOLD:
            pass


    def step(self, action):

        # Increment the step index.
        self.current_step += 1

        # If the step index has exceeded the data frame, then we're done.
        done = False
        if self.current_step >= len(self.df.index) - self._data_window:
            done = True


    def reset(self):
        self.balance = self._start_balance # Current account balance (i.e., spending money).
        self.shares = 0 # Current number of shares.
        self.cost_basis = 0 # Original value of asset for tax purposes.

        # Randomly initialize the step to be a point within the data frame.
        self.current_step = self.np_random.randint(
            low=0,
            high=len(self.df.index)-self._data_window,
            )


    def render(self, mode: str = 'human'):
        # Ideas for rendering:
        # - https://github.com/notadamking/Stock-Trading-Visualization/blob/master/render/StockTradingGraph.py
        pass