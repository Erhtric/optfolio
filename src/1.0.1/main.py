import numpy as np
# import cla
# import plot
import simplex

if __name__ == "__main__":

    A = np.array([
        [ 1, 3, 2, 1, 0],
        [ 1, 5, 1, 0, 1]])
    X = np.array([
        [-1, 1, 1, 0, 0],
        [ 1, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 1]])
    b = np.array([10, 8])
    c = np.array([8, 10, 7, 0, 0])

    #cla_pf = cla.CLA(['TSLA', 'GME', "AAPL"], [], [], '2014-01-01', '2016-12-31')
    #print(cla_pf)
    handler = simplex.Simplex(c, A, b)
    handler.simplex()
    handler.print_solution()