from typing import Tuple
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

if matplotlib.get_backend() == 'agg':
    print("matplotlib currently using 'agg' backend.\nAll the plots will be saved in ./pictures")
    save_fig = True

def f(x):
    z = 3*x**3 + 3*x**2 + 1
    return z

if __name__ == "__main__":
    x = np.linspace(-5, 5, 200)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(x, f(x))

    if save_fig: plt.savefig("./pictures/mygraph.png")