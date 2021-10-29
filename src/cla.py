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
        self.start_date = start_date
        self.end_date = end_date

        self.mean = self.compute_individual_mean()
        self.covariance = self.compute_covariance_matrix()

        if lower.shape[0] != len(self.tickers) or upper.shape[0] != len(self.tickers) or \
            any(l > 1.0 for l in lower) or any(u > 1.0 for u in upper):
            raise Exception("STOPPED EXECUTION: PORTFOLIO NOT INITIALIZED - PLEASE INSERT LEGIT BOUND ARRAYS")
        else:
            self.lb = lower.reshape((lower.shape[0], 1))
            self.ub = upper.reshape((upper.shape[0], 1))

        # WEIGTHS initialization
        self.weights = np.ones(len(tickers)).reshape((len(tickers), 1))
        self.weights = self.weights / self.weights.sum(axis=0, keepdims=True)

    def get_market_data(self, start_date, end_date):
        if not os.path.isfile(f'./src/data/ASSET_DATA_{start_date}_to_{end_date}_{self.tickers}.csv'):
            print('File not found, initializing download session')
            data.get_history_data(self.tickers, start_date, end_date)

        df = pd.read_csv(f'./src/data/ASSET_DATA_{start_date}_to_{end_date}_{self.tickers}.csv').set_index('formatted_date')
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
        return COV.cov().to_numpy()

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

    def compute_returns(self):
        """Compute the gain/loss on the portfolio over the fixed timeframe specified at initialization.
        """
        close_prices = pd.DataFrame()
        for count, ticker in enumerate(self.tickers):
            close_prices.insert(count, f"{ticker}", self.market_data['close'][self.market_data['ticker']==ticker])
        return close_prices.pct_change()#.iloc[1: , :]       # remove the first NaN row

    def compute_portfolio_expected_return(self):
        """Computes the expected portfolio's expected return defined as the weighted sum of the returns on the assets of the portfolio.
        """
        return self.compute_returns() @ self.weights

    def compute_portfolio_variance(self):
        print(type(self.weights))
        print(type(self.covariance))
        return self.weights.T @ self.covariance @ self.weights

    def compute_portfolio_std(self):
        """Computes the portfolio's standard deviation, also called the volatily of the portfolio"""
        return np.sqrt(self.compute_portfolio_variance())

    def to_standard_form(self):
        matrix = np.zeros((self.lb.shape[0] + self.ub.shape[0] + 2, len(self.tickers) + self.lb.shape[0] + self.ub.shape[0] + 1))\

        self.__set_bounds_coeff(matrix)
        self.__set_weights(matrix)
        # self.__set_utility_function_1(matrix)
        return matrix

    def __set_weights(self, matrix):
        """Set the FULL-INVESTMENT CONSTRAINT, it requires thath all the coefficients for the variables
        to be equal to 1.0.
        """
        for c in range(self.weights.shape[0]):
            matrix[-2, c] = 1
        matrix[-2, -1] = 1

    def __set_utility_function_1(self, matrix):
        """Set the utility function as specified by the MAXIMUM EXPECTED RETURN PORTFOLIO where
        the utility function is the following:
            u = exp_p
        with exp_p the expected return of the portfolio.
        """
        exp_p = self.compute_portfolio_expected_return().to_numpy()
        print(self.weights.shape)
        print(self.weights.T @ exp_p)

    def __set_bounds_coeff(self, matrix):
        """Set the constants for the upper and lower bound. We already know that in a standard allocation problem
        the bound equations are in the following form:
            - w0 >= lb0, w1 >= lb1, ..., wn >= lbn
            - w0 <= ub0, w1 <= ub1, ..., wn <= ubn
        """
        for r in range(self.lb.shape[0]):
            # The first m rows are for the lower bound (<=)
            # The constants are by definition all nonnegative!
            # To convert into equation we need to add a SLACK variable
            matrix[r, -1] = self.lb[r]
            matrix[r,  r] = 1   # for the variable
            matrix[r,  r + self.lb.shape[0]] = 1   # for the slack variable

            # The last m rows are for the upper bound (>=)
            # Since the RHS is always nonnegative we do not have to multiply by -1
            # To convert into equation we need to subtract a SURPLUS variable
            matrix[r + self.lb.shape[0], -1] = self.ub[r]
            matrix[r + self.lb.shape[0],  r] = 1    # for the variable
            matrix[r + self.lb.shape[0],  r + 2 * self.lb.shape[0]] = 1    # for the surplus variable
        return matrix

    def print_stats(self):
        print('Portfolio data:')
        print(f'\t-tickers: {self.tickers}')
        print(f'\t-participations: {list(self.weights[:, 0])}')
        print(f'\t-variance: {self.compute_portfolio_variance()}')
        print(f'\t-standard deviation (volatility): {self.compute_portfolio_std()}')
        print(f'Covariance matrix: \n{self.compute_covariance_matrix()}')



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