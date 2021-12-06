"""Generate a LaTeX table for the statistics of specifc stocks.
"""
import datetime
import os
import pandas as pd
import smart_stock as ss

# stocks = ['aapl', 'nvda', 'dis', 'ko', 'pg']
stocks = ['aapl', 'goog']
stocks_upper = list(s.upper() for s in stocks)

outdir = os.path.join(os.path.dirname(__file__), '..', 'latex', 'tables')
outfile = 'dataset_metrics_table.tex'
label = 'table:dataset_metrics'
caption = f"Statistics for Huge Stock Market Dataset stocks used for training."
outpath = os.path.join(outdir, outfile)

dataset_root = '~/Desktop'

def main():

    # Prepare dataset.
    path = os.path.expanduser(dataset_root)
    dataset = ss.datasets.HugeStockMarketDataset(path)

    # List of assets.
    dfs = {}
    for stock in stocks:
        df = dataset[stock][ss.envs.StockDataEnv.df_obs_cols]
        dfs[stock.upper()] = df.describe().loc[['min','max','mean','std']].transpose()

    # Collect all frames using stock name as index.
    df = pd.concat(dfs)

    # Dump frame as LaTeX.
    latex = df.to_latex(
        multirow=True,
        label=label,
        caption=caption,
        )
    latex = f"% GENERATED ON {datetime.datetime.now()}\n{latex}"
    with open(outpath, 'w') as f:
        f.write(latex)


if __name__ == '__main__':
    main()