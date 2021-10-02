import numpy as np
import pandas as pd
from data import *

class Portfolio:
    """
    A portfolio is made up of:
        - the assets oh which the portfolio is made, usually expressed by their tickers.
        - the initial amount to be invested (in $)
        - a list of weights, one for each asset considered. It represents the participation of the quotes in the portfolio.
        - a starting date and a final date which represent the temporal window for the historical data. Moreover a default 
            period of one day was chosen.
    """

    assets = []
    asset_data = None
    weights = [] 
    initial_investement = 0
    cov = 0

    def __init__(self, symbols, initial_amount, weights, start_date, end_date, period='1d'):
        self.assets = symbols

        # This is pretty delicate since it is the decision variable we have to minimize
        # Thus the variable initialized by the constructor is not a final assignment but 
        # an initialization for the descent methods to optimize.
        self.weights = weights
        #assert np.sum(weights) == 1
        print(60*'*')
        print("...Starting portfolio session...")

        self.initial_investment = initial_amount
        self.asset_data = data.get_history_data(symbols, start_time=start_date, end_time=end_date, period=period)

        # Output formatting
        print("Total number of assets in the portfolio:", len(self.assets))
        print("Initial amount invested:", self.initial_investement, "$")
        print("  Ticker\tInitial Weights")
        for i, a in enumerate(self.assets):
            print("  {0}\t\t{1}".format(a, weights[i]))
        print(60*'*')

        print('Additional information:')
        for sym in self.assets:
            df = self.asset_data[self.asset_data['Ticker'] == sym]
            print('Number of working days for {}: {}'.format(sym, df.shape[0]))


    def covariance(self) -> pd.DataFrame:
        """Computes the covariance for the already defined assets in the portfolio.

        Returns:
            cov: the covariance matrix (a pandas DataFrame) for the n assets
        """
        closing_prices = self.asset_data[['Close','Ticker']]
        sym_df = pd.DataFrame()
        for sym in self.assets:
            self.asset_data = closing_prices[closing_prices['Ticker'] == sym]
            sym_df.insert(loc=len(sym_df.columns), column=sym, value=self.asset_data['Close'])

        mean = np.mean(sym_df.to_numpy(), axis=0, dtype=np.float64)
        sym_df_demean = sym_df - mean
        m = sym_df.shape[0]
        self.cov = (sym_df_demean.T @ sym_df_demean) / m

        return self.cov

    def plot_ts_sep(self, attr):
        """Plot the data in separate plots for the given attribute"""
        data.plot_time_series(self.assets, self.asset_data, attr)


# pf = Portfolio(["GME", "AAPL", "TSLA", "AMC"], 1000, np.zeros(4), "2019-01-01", "2020-01-01")
#df = pf.asset_data[["Close", "Ticker"]]
# pf.plot_ts_sep('Close')
#print(df)
# pf.covariance()