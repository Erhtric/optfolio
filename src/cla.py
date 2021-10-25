import data
import pandas as pd
import numpy as np
import os

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

        if self.lb.shape[0] != len(self.tickers) or self.ub.shape[0] != len(self.tickers):
            raise Exception("STOPPED EXECUTION: PORTFOLIO NOT INITIALIZED - PLEASE INSERT LEGIT BOUND ARRAYS")
        else:
            self.lb = lower.reshape((lower.shape[0], 1))
            self.ub = upper.reshape((upper.shape[0], 1))

        self.weights = np.ones(len(tickers))

    def get_market_data(self, start_date, end_date):
        if not os.path.isfile(f'./src/data/ASSET_DATA_{start_date}_to_{end_date}.csv'):
            print('File not found, initializing download session')
            data.get_history_data(self.tickers, start_date, end_date)

        df = pd.read_csv(f'./src/data/ASSET_DATA_{start_date}_to_{end_date}.csv').set_index('formatted_date')
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

    def to_matrix_form(self):
        matrix = np.zeros((self.lb.shape[0] + self.ub.shape[0] + 2, len(self.tickers) + 1))\

        self.__set_bounds_coeff(matrix)
        return matrix

    def __set_weights(self, matrix):
        """
        """
        pass

    def __set_objective(self, matrix):
        """
        """
        pass

    def __set_bounds_coeff(self, matrix):
        """Set the constants for the upper and lower bound. We already know that in a standard allocation problem
        the bound equations are in the following form:
            - w0 >= lb0, w1 >= lb1, ..., wn >= lbn
            - w0 <= ub0, w1 <= ub1, ..., wn <= ubn
        """
        for r in range(self.lb.shape[0]):
            # The first m rows are for the lower bound (<=)
            matrix[r, -1] = self.lb[r]
            # The last m rows are for the upper bound (>=)
            # TODO: in some way we have to transform these diseq by multiplying them by -1
            matrix[r + self.lb.shape[0], -1] = self.ub[r]
            matrix[r + self.lb.shape[0], -1] = - matrix[r + self.lb.shape[0], -1]
        return matrix

    def set_utility_function(self):
        pass

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

    # def to_standard_form():
    #     pass