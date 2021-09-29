class Portfolio:

    def __init__(self, tickers, lower, upper):
        """
        Initializes a portfolio instance

        Args:
            mean ([type]): the expected return of the portfolio
            covariance ([type]): the covariance between assets in the portfolio
            lower ([type]): [description]
            upper ([type]): [description]
        """
        self.tickers = tickers
        self.data = 0       # stock data formatted as a Pandas DataFrame

        self.mean = compute_mean()
        self.covariance = compute_covariance()
        self.lb = lower
        self.ub = upper

        self.weights = []

        # For the Constrained Optimization problem
        self.lambdas = []
        self.etas = []

        def compute_mean(self):
            pass

        def compute_covariance(self):
            pass

class CLA(Portfolio):

    def __init__(self) -> None:
        """
        Initializes a CLA instance 
        """
        super()

    def solve(self):
        pass