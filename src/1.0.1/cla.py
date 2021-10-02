import data
import pandas as pd
import numpy as np

class Portfolio:

    def __init__(self, tickers, lower, upper, start_date, end_date):
        """
        Initializes a portfolio instance

        Args:
            mean ([type]): the expected return of the portfolio
            covariance ([type]): the covariance between assets in the portfolio
            lower ([type]): [description]
            upper ([type]): [description]
        """
        self.tickers = tickers
        self.market_data = self.get_market_data(start_date, end_date)       # stock data formatted as a Pandas DataFrame

        self.mean = self.compute_individual_exp()
        self.covariance = self.compute_covariance_matrix()
        self.lb = lower
        self.ub = upper

        self.weights = []

        # For the Constrained Optimization problem
        self.lambdas = []
        self.etas = []

    def get_market_data(self, start_date, end_date):
        data.get_history_data(self.tickers, start_date, end_date)
        df = pd.read_csv('./src/1.0.1/data/ASSET_DATA.csv')
        return df

    def compute_individual_exp(self):
        """
        Computes the vector of the individual assets' expected returns
        """
        mean = np.zeros(len(self.tickers))
        for count, ticker in enumerate(self.tickers):
            close = self.market_data['close'][self.market_data['ticker']==ticker]
            mean[count] = np.mean(close)
        return mean

    def compute_portfolio_exp(self):
        """
        Computes the portfolio's expected return
        """
        return np.mean(self.mean, axis=0)

    def compute_covariance_matrix(self):
        close = self.market_data[['close', 'ticker']]

        # Combine the stock data in a single matrix where each column has the prices for the individual asset
        S = pd.DataFrame()
        

pf = Portfolio(['TSLA', 'GME', "AAPL"], [], [], '2014-01-01', '2016-12-31')
print(pf.compute_covariance())

class CLA(Portfolio):

    def __init__(self) -> None:
        """
        Initializes a CLA instance 
        """
        super()

    def solve(self):
        pass