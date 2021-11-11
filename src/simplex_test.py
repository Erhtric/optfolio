import numpy as np
from solvers.simplex import Simplex
from portfolio import Portfolio
#np.set_printoptions( threshold=20, edgeitems=10, linewidth=140, formatter = dict( float = lambda x: "%.4g" % x ))  # float arrays %.3g

def risk_function(self, cov, x):
    return x.T @ cov @ x

if __name__ == "__main__":

    # # max z = c.T @ x
    # c = np.array([4, 5, 6, 0, 0, 0])
    # # s.t A @ x <= b
    # A = np.array([[ 2, 3, 1, 1, 0, 0],
    #               [ 3, 1, 1, 0, 1, 0],
    #               [ 4, 2, 1, 0, 0, 1]
    #     ])
    # b = np.array([900, 350, 400])

    # # max z = c.T @ x
    # c = np.array([1, 1, 0, 0, 0])
    # # s.t A @ x <= b
    # A = np.array([[ -1, 1, 1, 0, 0],
    #               [  1, 0, 0, 1, 0],
    #               [  0, 1, 0, 0, 1]
    #     ])
    # b = np.array([2, 4, 4])

    # # min z = c.T @ x = max -z
    # c = np.array([-2, -3, -4, 0, 0])
    # # s.t A @ x <= b
    # A = np.array([[3, 2, 1, 1, 0],
    #               [2, 5, 3, 0, 1]
    #     ])
    # b = np.array([10, 15])

    tickers = ['TSLA', 'GME', 'AAPL', 'JNJ', 'SPCE']
    pf = Portfolio(tickers
                    , np.zeros(len(tickers))
                    , np.array([0.3, 0.6, 1.0, 0.6, 1.0])
                    , '2020-01-01', '2021-01-01')

    # TEST FOR THE MAXIMUM EXPECTED RETURN PORTFOLIO - linear programming
    pf.solve_simplex_LP(False)
    pf.print_stats()

    # TEST FOR THE MEAN VARIANCE PORTFOLIO - quadratic programming
