from numpy.core.fromnumeric import argsort
import data
import pandas as pd
import numpy as np

class Portfolio:

    def __init__(self, tickers, lower, upper, start_date, end_date):
        """
        Initializes a portfolio instance

        Args:
            tickers: a list of assets to be included in the portfolio
            lower: numpy array describing the lower bounds
            upper: numpy array describing the upper bounds
            start_date: the starting date for the historical data
            end_date: the ending date for the historical data
        """
        self.tickers = tickers
        self.market_data = self.get_market_data(start_date, end_date)       # stock data formatted as a Pandas DataFrame

        self.mean = self.compute_individual_mean()
        self.covariance = self.compute_covariance_matrix()
        self.lb = lower
        self.ub = upper

        self.weights = np.zeros(len(tickers))

    def get_market_data(self, start_date, end_date):
        data.get_history_data(self.tickers, start_date, end_date)
        df = pd.read_csv('./src/1.0.1/data/ASSET_DATA.csv').set_index('formatted_date')
        return df

    def compute_individual_mean(self):
        """
        Computes the vector of the individual assets' means
        """
        mean = np.zeros(len(self.tickers))
        for count, ticker in enumerate(self.tickers):
            close = self.market_data['close'][self.market_data['ticker']==ticker]
            mean[count] = close.mean()
        return mean

    def compute_portfolio_mean(self):
        """
        Computes the portfolio's mean
        """
        return np.mean(self.mean, axis=0)

    def compute_covariance_matrix(self):
        """
        Compute the covariance matrix between the assets
        """
        # Combine the stock data in a single matrix where each column has the prices for the individual asset
        COV = pd.DataFrame()
        for count, ticker in enumerate(self.tickers):
            close_prices = self.market_data['close'][self.market_data['ticker']==ticker]
            COV.insert(count, ticker, close_prices, allow_duplicates=True)
        return COV.cov()

    def compute_correlation_matrix(self):
        """
        Compute the correlation matrix between the assets
        """
        # Combine the stock data in a single matrix where each column has the prices for the individual asset
        COV = pd.DataFrame()
        for count, ticker in enumerate(self.tickers):
            close_prices = self.market_data['close'][self.market_data['ticker']==ticker]
            COV.insert(count, ticker, close_prices, allow_duplicates=True)
        return COV.corr()

    def compute_portfolio_expected_return(self):
        return self.mean @ self.weights

    def compute_portfolio_variance(self):
        return self.weights.T @ (self.covariance @ self.weights)

    def compute_portfolio_std(self):
        return np.sqrt(self.compute_portfolio_variance())

class CLA(Portfolio):

    def __init__(self, tickers, lower, upper, start_date, end_date) -> None:
        """
        Initializes a CLA instance
        """
        super().__init__(tickers, lower, upper, start_date, end_date)

        # For the Constrained Optimization problem
        self.lambdas = []
        self.etas = []


    def solve(self):
        """
        Solving method for the COP. Such problem in our context consists in
        a optimization procedure where we want to optimize an utility function.
        For example, if we want the maximum expected return we would write:
                max     expected return on the portfolio \\
                where   risk_tolerance -> infinity \\
                s.t.    weights.T @ 1 = weights \\
                        A @ weights = b \\
                        weigths >= lb \\
                        weigths <= ub
        Such problems could be solved by a general linear programming algorithm.
        """
        pass

cla_pf = CLA(['TSLA', 'GME', "AAPL"], [], [], '2014-01-01', '2016-12-31')
