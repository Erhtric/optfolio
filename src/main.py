import numpy as np
# import cla
# import plot
from solvers.simplex import Simplex
import solvers.cg_gradient

from portfolio import Portfolio

if __name__ == "__main__":

    # A = np.array([
    #     [ 1, 3, 2, 1, 0],
    #     [ 1, 5, 1, 0, 1]])
    # b = np.array([10, 8])
    # c = np.array([8, 10, 7, 0, 0])

    # c = np.array([1, 1, 0, 0, 0])
    # A = np.array([
    #     [-1, 1, 1, 0, 0],
    #     [ 1, 0, 0, 1, 0],
    #     [ 0, 1, 0, 0, 1]])
    # b = np.array([2, 4, 4])

    # c = np.array([1, 1, 0])
    # A = np.array([
    #     [-1, 4, 1]])
    # b = np.array([13])

    #cla_pf = cla.CLA(['TSLA', 'GME', "AAPL"], [], [], '2014-01-01', '2016-12-31')
    #print(cla_pf)
    #handler = simplex.Simplex(c, A, b, True)
    #handler.simplex()
    #handler.print_solution()

    tickers = ['TSLA', 'GME', 'AAPL', 'JNJ']
    pf = Portfolio(tickers, np.array([0.1, 0.0, 0.0, 0.0]), np.array([0.3, 0.3, 1.0, 1.0]), '2020-01-01', '2021-01-01')
    #print(pf.to_standard_form())
    # pf.print_stats()
    print(pf.compute_returns_mean())
