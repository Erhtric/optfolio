import numpy as np
# import cla
# import plot
from solvers.simplex import Simplex
import solvers.cg_gradient

from portfolio import Portfolio

if __name__ == "__main__":

    # tickers = ['TSLA', 'GME', 'AAPL', 'JNJ']
    tickers = ['TSLA', 'GME', 'AAPL']
    # pf = Portfolio(tickers, np.array([0.1, 0.0, 0.0, 0.0]), np.array([0.3, 0.3, 1.0, 1.0]), '2020-01-01', '2021-01-01')
    pf = Portfolio(tickers, lower=np.array([0.1, 0.0, 0.3]), upper=np.array([0.5, 1.0, 1.0]), start_date='2020-01-01', end_date='2021-01-01')
    pf.solve_simplex()
    pf.print_stats()
    # print(pf.to_standard_form())