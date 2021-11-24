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
        1       Net worth       -Inf     Inf
        2       Shares          0.       Inf
        3       Open            0.       Inf
        4       High            0.       Inf
        5       Low             0.       Inf
        6       Close           0.       Inf
        7       Volume          0.       Inf

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
    df_obs_cols = ['Open','High','Low','Close','Volume']

    def __init__(self, 
        df: DataFrame,
        start_balance: float,
        max_stock: int = 100,
        # max_steps: int = None,
        start_day: int = None,
    ):
        super().__init__()

        # Initial reset parameters.
        self._start_day = start_day
        self._start_balance = start_balance

        # if max_steps is None:
        #     self._max_steps = len(df.index)
        # else:
        #     self._max_steps = max_steps

        # Preserve the input dataframe.
        self.df = df

        # Define action space.
        self._max_stock = max_stock
        self.action_space = gym.spaces.Box(
            low=-max_stock,
            high=max_stock,
            shape=(1,),
            dtype=np.int64,
        )

        # Define state space.
        self.observation_space = gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(8,),
            dtype=np.float32,
        )

        # Seed the environment.
        self.seed()


    def seed(self, seed=None):
        self.action_space.seed(seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def _action_buy(self, 
        shares: int,
        stock_price: float,
        ):

        # Compute total purchase cost and cost basis for tax purposes.
        cost = shares * stock_price
        self.balance -= cost
        if self.shares + shares > 0:
            self.cost_basis = (cost + self.cost_basis*self.shares)/(self.shares + shares)

        # Update number of shares held.
        self.shares += shares


    def _action_sell(self, 
        shares: int,
        stock_price: float,
        ):

        # # Cap number of shares to sell based on our inventory.
        # shares = min(self.shares, shares)

        # Compute sale price and update balance.
        self.balance += shares * stock_price

        # Update current shares.
        self.shares -= shares


    def _perform_action(self, action):

        # Set current stock price at a random value 
        # between the start and end price for the day.
        stock_price = self.np_random.uniform(
            low=self.df['Open'].iloc[self.current_step],
            high=self.df['Close'].iloc[self.current_step],
        )

        # Perform buy action.
        if action > 0:
            shares = action
            self._action_buy(shares, stock_price)

        # Perform sell action.
        elif action < 0:
            shares = np.abs(action) # Convert number of stocks to sell to be a positive number.
            self._action_sell(shares, stock_price)

        # Perform hold action.
        # This is most likely "do nothing".
        else:
            pass

        # Update current net worth.
        self.net_worth = self.balance + (self.shares * stock_price)


    def _get_observation(self):

        # Get necessary rows of data frame as numpy matrix.
        obs = self.df[self.df_obs_cols].iloc[self.current_step].to_numpy()

        # Prepend agent-specific observations.
        # Adds:
        # - Balance
        # - Net worth
        # - Shares
        obs = np.concatenate(([self.balance, self.net_worth, self.shares], obs))

        return obs


    def step(self, action):

        # Preserve current net worth.
        curr_net_worth = self.net_worth

        # Take the given action.
        self._perform_action(action)

        # Increment the step index.
        self.current_step += 1

        # If the step index has exceeded the data frame, then we're done.
        done = False
        if self.current_step >= len(self.df.index):
            done = True

        # Agent has run out of money.
        if self.balance <= 0:
            done = True

        # Compute reward.
        # Reward is the current balance multiplied by
        # a fraction of the current step given the maximum
        # number of steps possible.
        # reward = self.balance * (self.current_step / self._max_steps)
        reward = (self.net_worth - curr_net_worth) * (2. ** -11.)

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
                high=len(self.df.index)-1,
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

        # Get current observation.
        obs = self._get_observation()

        # Human readable text.
        if mode is None or mode == RenderMode.ASCII:
            print(f"[{self.current_step}] Balance: {obs[0]}")
            print(f"[{self.current_step}] Net Worth: {obs[1]}")
            print(f"[{self.current_step}] Shares: {obs[2]}")
            print(f"[{self.current_step}] Open: {obs[3]}")
            print(f"[{self.current_step}] High: {obs[4]}")
            print(f"[{self.current_step}] Low: {obs[5]}")
            print(f"[{self.current_step}] Close: {obs[6]}")
            print(f"[{self.current_step}] Volume: {obs[7]}")

        # CSV format.
        elif mode == RenderMode.CSV:
            # items = [self.current_step, self.balance, self.shares, self.net_worth, self.cost_basis]
            print(','.join(str(x) for x in obs))