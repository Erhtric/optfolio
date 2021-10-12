import numpy as np
import cla
import plot
import simplex

if __name__ == "__main__":

    #cla_pf = cla.CLA(['TSLA', 'GME', "AAPL"], [], [], '2014-01-01', '2016-12-31')
    #print(cla_pf)

    A = np.array([
        [-1, 1, 1, 0, 0],
        [ 1, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 1]])
    b = np.array([
        [2, 4, 5]
    ])
    c = np.array([1, 1, 0, 0, 0])
    xb = np.array([np.concatenate([eq, bb]) for eq, bb in zip(A, b.T)])
    z = np.concatenate((c, [0])).reshape((xb.shape[1],1))
    print(np.concatenate((xb, z.T), axis=0))