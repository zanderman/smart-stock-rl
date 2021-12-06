# Jupyter Notebooks

These Jupyter notebooks can be opened in Google Colab:

- [experiments_qsfm.ipynb](./experiments_qsfm.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zanderman/smart-stock-rl/blob/main/jupyter/experiments_qsfm.ipynb)
- [experiments_dqn.ipynb](./experiments_dqn.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zanderman/smart-stock-rl/blob/main/jupyter/experiments_dqn.ipynb)

## Runtime Guide

To run these experiments you will need a Kaggle API token to download the stock market dataset (see instructions at <https://www.kaggle.com/docs/api>).

The Jupyter notebooks accept Kaggle API token input via 3 methods:

1. (Best for local runtime) `~/.kaggle/kaggle.json`
2. (Best for Google Colab) Copy/paste the contents of the entire API token JSON blob into the Jupyter notebook when prompted
3. Type your Kaggle username and API token into the Jupyter notebook when prompted

Note that this authentication is only required on the first run. The notebooks preserve this information across environment restarts for ease of use.
