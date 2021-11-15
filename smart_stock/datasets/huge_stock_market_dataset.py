
import glob
import kaggle
import os
import pandas as pd


class HugeStockMarketDataset:
    """Wrapper for Huge Stock Market Dataset by Boris Marjanovic on Kaggle.

    https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3
    """
    root = 'huge_stock_market_dataset'

    def __init__(self, 
        path: str, 
        files: list = None,
        ):
        self._index = {}

        # Download the dataset if necessary.
        newpath = os.path.join(path, self.root)
        if not os.path.exists(newpath):
            self.download(newpath, files)
        else:
            self.path = newpath
            self._build_index()


    def __getitem__(self, item):
        if isinstance(item, str):
            return self.get_dataframe(item)
        elif isinstance(item, list):
            if len(item) == 1:
                return self.get_dataframe(item[0])
            else:
                return [self.get_dataframe(ticker) for ticker in item]


    def __len__(self):
        return len(self._index)


    @property
    def stocks(self) -> list:
        """Returns a list of all downloaded stocks and ETFs."""
        return list(self._index.keys())


    def _build_index(self):
        """Creates an internal index of stocks and ETFs for lookup."""
        for file in glob.iglob(os.path.join(self.path, '**', '*.txt'), recursive=True):
            filename = os.path.basename(file)
            stock_name = filename.split('.')[0]
            self._index[stock_name] = file


    def download(self, path: str, files: list = None):
        """Downloads the dataset from Kaggle.

        Args:
            path (str): The path to place the download.
            files (list, optional): Subset list of files to download instead of entire dataset. Defaults to None.
        """

        kaggle_dataset = 'borismarjanovic/price-volume-data-for-all-us-stocks-etfs'

        kaggle.api.authenticate()

        if files is not None:
            for f in files:
                kaggle.api.dataset_download_file(
                    dataset=kaggle_dataset,
                    file_name=f,
                    path=os.path.join(path, *os.path.split(f)),
                )
        else:
            kaggle.api.dataset_download_files(
                dataset=kaggle_dataset,
                path=path,
                unzip=True,
            )

        # Save the new downloaded path.
        self.path = path

        # Force rebuild the index after downloading.
        self._build_index()


    def get_dataframe(self, ticker: str) -> pd.DataFrame:
        """Obtain historical data for stock or ETF in a pandas dataframe.

        Args:
            ticker (str): The identifier for the stock or ETF.

        Returns:
            pd.DataFrame: Historical data.
        """
        return pd.read_csv(self._index[ticker])