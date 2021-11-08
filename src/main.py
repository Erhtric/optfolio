import numpy as np
from solvers.simplex import Simplex
from solvers.interior_point import IntPoint
from portfolio import Portfolio
#np.set_printoptions( threshold=20, edgeitems=10, linewidth=140, formatter = dict( float = lambda x: "%.4g" % x ))  # float arrays %.3g

if __name__ == "__main__":

    """c = np.array([4, 1, 4, 0, 0, 0])
    A = np.array([  [ 2, 1, 1, 1, 0, 0],
                    [ 1, 2, 3, 0, 1, 0],
                    [ 2, 2, 1, 0, 0, 1]])
    b = np.array([2, 4, 8])
    """
    """c = np.array([15, 10, 0, 0, 0])
    A = np.array([
        [  0.25, 1, 1, 0, 0],
        [  1.25, 0.5, 0, 1, 0],
        [  1, 1, 0, 0, 1]])
    b = np.array([65, 90, 85])"""

    c = np.array([-2, -3, -4, 0, 0])
    A = np.array([  [ 3, 2, 1, 1, 0],
                    [ 2, 5, 3, 0, 1],
                    ])
    b = np.array([10, 15])

    #handler = Simplex(c, A, b, verbose=True)
    #handler.solve()
    #handler.print_solution()
    #handler.plot_objective_function()

    # tickers = ['TSLA', 'GME', 'AAPL', 'JNJ']
    tickers = ['TSLA', 'GME']
    #pf = Portfolio(tickers, np.array([0.1, 0.0, 0.0, 0.0]), np.array([0.3, 0.3, 1.0, 1.0]), '2020-01-01', '2021-01-01')
    pf = Portfolio(tickers, lower=np.array([0.0, 0.0]), upper=np.array([1.0, 1.0]), start_date='2020-01-01', end_date='2021-01-01')
    #pf.solve_simplex(True)
    #pf.print_stats()

    cov = pf.compute_returns_covariance_matrix()
    S, A, b = pf.set_matrix_qp()
    weights = np.array([0.5, 0.5])

    n = A.shape[1]
    m = A.shape[0]
    init_x = np.ones(n)
    init_y = 1*np.ones(m)
    init_l = 2*np.ones(m)

    ip = IntPoint(S, A, b, 0)
    ip.corrector_step(init_x, init_y, init_l, np.zeros((m,m)), np.zeros((m,m)), 0)