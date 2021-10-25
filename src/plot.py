from typing import List
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def plot_time_series(symbols: List[str], attr: str):
    """Simple plotting utility function to plot the timeseries for the given attribute.

    Args:
        symbols (List[str]): the list of symbols we want to plot
        attr (str): the attribute we want to focus on
    """
    colors = plt.rcParams["axes.prop_cycle"]()  # get the color cycler
    n_symbols = len(symbols)
    columns = 2
    rows = int(np.ceil(n_symbols / 2))

    DATA = pd.read_csv('./src/data/ASSET_DATA.csv')

    fig = plt.figure()
    fig.suptitle(f'{attr} prices ($)')
    for count in range(1, n_symbols+1):
        ticker = symbols[count-1]
        date = DATA['formatted_date'][DATA['ticker']==ticker]
        values = DATA[attr][DATA['ticker']==ticker]
        ax = fig.add_subplot(rows, columns, count)
        ax.set_title(symbols[count-1])
        ax.set_xticks([])

        # Get the next color from the cycler
        c = next(colors)["color"]
        plt.plot(date, values, color=c)
        plt.grid(True)

    fig.tight_layout()
    fig.savefig(f'./src/results/{attr}_time_series.pdf')
    fig.show()