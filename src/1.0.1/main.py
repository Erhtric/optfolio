import numpy as np
import cla
import plot
import simplex

if __name__ == "__main__":

    A = np.array([
        [-1, 1, 1, 0, 0],
        [ 1, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 1]])
    b = np.array([
        [2, 4, 5]
    ])
    c = np.array([1, 1, 0, 0, 0])

    #cla_pf = cla.CLA(['TSLA', 'GME', "AAPL"], [], [], '2014-01-01', '2016-12-31')
    #print(cla_pf)
    handler = simplex.Simplex(c, A, b)