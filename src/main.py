import numpy as np
# import cla
# import plot
from solvers.simplex import Simplex
from portfolio import Portfolio
np.set_printoptions( threshold=20, edgeitems=10, linewidth=140,
    formatter = dict( float = lambda x: "%.3g" % x ))  # float arrays %.3g

if __name__ == "__main__":

    """
    c = np.array([4, 1, 4, 0, 0, 0])
    A = np.array([  [ 2, 1, 1, 1, 0, 0],
                    [ 1, 2, 3, 0, 1, 0],
                    [ 2, 2, 1, 0, 0, 1]])
    b = np.array([2, 4, 8])"""

    c = np.array([13, 23, 0, 0, 0])
    A = np.array([
        [  5, 15, 1, 0, 0],
        [  4,  4, 0, 1, 0],
        [ 35, 20, 0, 0, 1]])
    b = np.array([480, 160, 1190])

    # tickers = ['TSLA', 'GME', 'AAPL', 'JNJ']
    tickers = ['TSLA', 'GME', 'AAPL']
    #pf = Portfolio(tickers, np.array([0.1, 0.0, 0.0, 0.0]), np.array([0.3, 0.3, 1.0, 1.0]), '2020-01-01', '2021-01-01')
    pf = Portfolio(tickers, lower=np.array([0.0, 0.0, 0.5]), upper=np.array([1.0, 1.0, 1.0]), start_date='2020-01-01', end_date='2021-01-01')
    pf.solve_simplex()
    #pf.print_stats()
    #print(pf.to_standard_form())