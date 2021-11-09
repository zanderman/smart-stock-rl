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
        max_steps: int = None,
    ):
        super().__init__()

        # Ensure that the length of the data is longer than the provided window.
        if len(df.index) <= history+1:
            raise ValueError('not enough data for history window size')

        # Initial reset parameters.
        self._history = history
        self._data_window = self._history + 1
        self._start_balance = start_balance
        if max_steps is None:
            self._max_steps = len(df.index)
        else:
            self._max_steps = max_steps

        self.df = df
        self.action_space = gym.spaces.Box(
            low=np.array([0., 0.]),
            high=np.array([3., 1.]),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=0.,
            high=np.finfo(np.float32).max,
            shape=(self._data_window,5,),
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
        action_type = ActionType(int(action[0]))
        action_percent_amount = action[1]

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
        obs = self.df[['Open','High','Low','Close','Volume']].iloc[self.current_step:self.current_step+self._data_window].to_numpy()

        return obs


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

        # Randomly initialize the step to be a point within the data frame.
        self.current_step = self.np_random.randint(
            low=0,
            high=len(self.df.index)-self._data_window,
            )

        # Get initial observation and return.
        obs = self._get_observation()
        return obs


    def render(self, mode: str = 'human'):
        # Ideas for rendering:
        # - https://github.com/notadamking/Stock-Trading-Visualization/blob/master/render/StockTradingGraph.py

        items = [self.current_step, self.balance, self.shares, self.net_worth, self.cost_basis]
        print(','.join(str(x) for x in items))