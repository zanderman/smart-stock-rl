
import os
import smart_stock as ss

path = os.path.expanduser('~/Desktop')
dataset = ss.datasets.HugeStockMarketDataset(path)


df = dataset['aapl']
start_balance = 100
history = 5
env = ss.envs.StockEnv(df, start_balance, history)

obs = env.reset()

action = env.action_space.sample()

next_obs, reward, done, info = env.step(action)

print('env.observation_space',env.observation_space)
print('obs',obs)
print('action', action.shape, action)
print('next_obs', next_obs.shape, next_obs)
print('reward', reward)
print('done', done)
print('info', info)