import numpy as np
# import cla
# import plot
import simplex
from cla import Portfolio

if __name__ == "__main__":

    # A = np.array([
    #     [ 1, 3, 2, 1, 0],
    #     [ 1, 5, 1, 0, 1]])
    # b = np.array([10, 8])
    # c = np.array([8, 10, 7, 0, 0])

    c = np.array([1, 1, 0, 0, 0])
    A = np.array([
        [-1, 1, 1, 0, 0],
        [ 1, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 1]])
    b = np.array([2, 4, 4])

    # c = np.array([1, 1, 0])
    # A = np.array([
    #     [-1, 4, 1]])
    # b = np.array([13])

    #cla_pf = cla.CLA(['TSLA', 'GME', "AAPL"], [], [], '2014-01-01', '2016-12-31')
    #print(cla_pf)
    #handler = simplex.Simplex(c, A, b, True)
    #handler.simplex()
    #handler.print_solution()

    tickers = ['TSLA', 'GME', 'AAPL']
    pf = Portfolio(tickers, np.zeros(len(tickers)), np.array([0.5, 0.5, 1.0]), '2020-01-01', '2021-01-01')
    print(pf.to_matrix_form())