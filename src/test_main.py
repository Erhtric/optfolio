import numpy as np
from solvers.simplex import Simplex
from solvers.interior_point import IntPoint
from portfolio import Portfolio
#np.set_printoptions( threshold=20, edgeitems=10, linewidth=140, formatter = dict( float = lambda x: "%.4g" % x ))  # float arrays %.3g

if __name__ == "__main__":


    # min z = c.T @ x = max -z
    c = np.array([-2, -3, -4, 0, 0])
    # s.t A @ x <= b
    A = np.array([[3, 2, 1, 1, 0],
                  [2, 5, 3, 0, 1]
        ])
    b = np.array([10, 15])

    slex = Simplex(c, A, b, verbose=False, max=False)
    slex.solve()
    slex.print_solution()

    print('\n\n')

    #min z = x.T @ S @ x + c.T @ x + const
    S = np.array([
        [ 1, -1],
        [-1,  2]
    ])
    c = np.array([-2, -6])
    # s.t. Ax>=b
    A = np.array([
        [-0.5, -0.5],
        [   1,   -2],
        [   1,    0],
        [   0,    1]
    ])
    b = np.array([-1, -2, 0, 0])

    intp = IntPoint(S, c, A, b
                    , x_init=np.ones(A.shape[1])
                    , y_init=np.full((A.shape[0],), 1.0)
                    , lm_init=np.full((A.shape[0],), 2.0)
                    , verbose=False)
    intp.solve()
    intp.print_solution()
