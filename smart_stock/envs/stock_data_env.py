import enum
from typing import Tuple
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


class RenderMode(enum.Enum):
    ASCII = 'ascii'
    CSV = 'csv'


class StockDataEnv(gym.Env):
    """Stock Environment with dataframe input.

    This environment mimics the behavior of a stock using
    its historical data for training/evaluation. Data must be provided in a pandas dataframe using proper headers.

    Observation:
        Type: Box(7, float)
        Num     Observation     Min      Max
        0       Balance         -Inf     Inf
        1       Shares          0.       Inf
        2       Open            0.       Inf
        3       High            0.       Inf
        4       Low             0.       Inf
        5       Close           0.       Inf
        6       Volume          0.       Inf

    Actions:
        Type: Box(1, int)
        Num     Action                                      Min     Max
        0       Integer number of stocks to buy/sell/hold   -k      k
                Hold = 0, Buy > 0, Sell < 0

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
    metadata = {'render.modes': [RenderMode.ASCII, RenderMode.CSV]}

    # Static list of header column names to use for each observation.
    _df_obs_cols = ['Open','High','Low','Close','Volume']

    def __init__(self, 
        df: DataFrame,
        start_balance: float,
        max_stock: int = 100,
        max_steps: int = None,
        start_day: int = None,
    ):
        super().__init__()

        # Initial reset parameters.
        self._start_day = start_day
        self._start_balance = start_balance
        if max_steps is None:
            self._max_steps = len(df.index)
        else:
            self._max_steps = max_steps

        # Preserve the input dataframe.
        self.df = df

        # Define action space.
        self.action_space = gym.spaces.Box(
            low=-max_stock,
            high=max_stock,
            dtype=np.int64,
        )

        # Define state space.
        self.observation_space = gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1,7,),
            dtype=np.float32,
        )

        # Seed the environment.
        self.seed()


    def seed(self, seed=None):
        self.action_space.seed(seed)
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
        self.balance -= cost
        if self.shares + bought_shares > 0:
            self.cost_basis = (cost + self.cost_basis*self.shares)/(self.shares + bought_shares)

        # Update number of shares held.
        self.shares += bought_shares


    def _action_sell(self, 
        stock_price: float, 
        action_percent_amount: float,
        ):

        # Number of shares to be sold.
        sold_shares = int(self.shares * action_percent_amount)

        # Compute sale price and update balance.
        self.balance += sold_shares * stock_price

        # Update current shares.
        self.shares -= sold_shares


    def _perform_action(self, action):

        # Set current stock price at a random value 
        # between the start and end price for the day.
        stock_price = self.np_random.uniform(
            low=self.df['Open'].iloc[self.current_step],
            high=self.df['Close'].iloc[self.current_step],
        )

        # Split the action into type and percentage amount.
        action_type, action_percent_amount = self.parse_action(action)

        # Perform buy action.
        if action_type == ActionType.BUY:
            self._action_buy(stock_price, action_percent_amount)

        # Perform sell action.
        elif action_type == ActionType.SELL:
            self._action_sell(stock_price, action_percent_amount)

        # Perform hold action.
        # This is most likely "do nothing".
        elif action_type == ActionType.HOLD:
            pass

        # Update current net worth.
        self.net_worth = self.balance + (self.shares * stock_price)


    def _get_observation(self):

        # Get necessary rows of data frame as numpy matrix.
        obs = self.df[self._df_obs_cols].iloc[self.current_step:self.current_step+self._data_window].to_numpy()

        return obs


    def parse_action(self, action) -> Tuple[ActionType, float]:
        """Splits an action into type and percentage amount."""
        action_type = ActionType(int(action[0]))
        action_percent_amount = action[1]
        return action_type, action_percent_amount


    def step(self, action):

        # Take the given action.
        self._perform_action(action)

        # Increment the step index.
        self.current_step += 1

        # If the step index has exceeded the data frame, then we're done.
        done = False
        if self.current_step >= len(self.df.index) - self._data_window:
            done = True

        # Agent has run out of money.
        if self.balance <= 0:
            done = True

        # Compute reward.
        # Reward is the current balance multiplied by
        # a fraction of the current step given the maximum
        # number of steps possible.
        reward = self.balance * (self.current_step / self._max_steps)

        # Get next observation.
        obs = self._get_observation()

        return obs, reward, done, {}


    def reset(self):
        self.balance = self._start_balance # Current account balance (i.e., spending money).
        self.shares = 0 # Current number of shares.
        self.cost_basis = 0 # Original value of asset for tax purposes.
        self.net_worth = 0

        # If no start day was given, then randomly initialize the step to be a point within the data frame.
        if self._start_day is None:
            self.current_step = self.np_random.randint(
                low=0,
                high=len(self.df.index)-self._data_window,
                )
        else:
            self.current_step = self._start_day

        # Get initial observation and return.
        obs = self._get_observation()
        return obs


    def render(self, mode: str = 'ascii'):
        # Ideas for rendering:
        # - https://github.com/notadamking/Stock-Trading-Visualization/blob/master/render/StockTradingGraph.py

        # Convert render mode string into enum (enforces validation too).
        mode = RenderMode(mode)

        # Human readable text.
        if mode == RenderMode.ASCII:
            print(f"[{self.current_step}] Balance: {self.balance}")
            print(f"[{self.current_step}] Shares: {self.shares}")
            print(f"[{self.current_step}] Net Worth: {self.net_worth}")
            print(f"[{self.current_step}] Cost Basis: {self.cost_basis}")

        # CSV format.
        elif mode == RenderMode.CSV:
            items = [self.current_step, self.balance, self.shares, self.net_worth, self.cost_basis]
            print(','.join(str(x) for x in items))