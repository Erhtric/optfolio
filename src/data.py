from typing import List
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

def get_history_data(symbols: list[str], start_time: str, end_time: str, period: str, save_file: bool=True) -> pd.DataFrame:
    """The method downloads the history of data from finance.yahoo.com
    of the given symbols, for a certain interval of time, and optionally
    it could save the sudden data to file.

    Args:
        symbols: the objective symbols
        start_time: start of the time-interval
        end_time: end of the time-interval
        period: specify a frequency for the downloaded data
        save_file (bool, optional): Trigger to save the file. Defaults to True.

    Returns:
        pandas.DataFrame: the dataset containing the data associated with the symbols
    """

    frames = []
    for ticker in symbols:
        tickerData = yf.Ticker(ticker)
        tickerDf = tickerData.history(start=start_time, end=end_time, period=period)
        tickerDf['Ticker'] = ticker
        if save_file: tickerDf.to_csv('./src/data/{}.csv'.format(ticker))
        frames.append(tickerDf)

    df = pd.concat(frames)

    # Clean the dataframe from the values in which we are not interested.
    df = df.drop(['Stock Splits'], axis=1)
    return df

def get_info_data(symbols: list[str], keyword: str) -> List:
    """The method get the relative info dictated with the relative keyword
        about the specific asset. See yfinance for further details or simply print
        the list of all related info by doing ticker.info.

    Args:
        keyword: the keyword used to get the information

    Returns:
        List: the list of values extracted
    """
    values = []
    for ticker in symbols:
        dict_info = yf.Ticker(ticker).info
        values.append(dict_info[keyword])

    return values

def plot_time_series(symbols: list[str], attr: str):
        """The method plots the time series for EACH ticker in the already selected
        chosen date interval.

        Args:
            attr: the attribute of the Dataframe to plot, it is possbile to choose one of those:
                Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
        """
        print("...Plotting \"{}\" for the following tickers: {}...".format(attr, symbols))

        colors = plt.rcParams["axes.prop_cycle"]()  # get the color cycler
        n = len(symbols)
        columns = 2 if n >= 2 else n
        rows = int(np.ceil(n / 2))

        fig = plt.figure()
        fig.suptitle(attr + ' prices ($)')
        for count in range(1, n+1):
            df = pd.read_csv('./src/data/{}.csv'.format(symbols[count-1]))
            # Convert the Date column to the datetime type.
            df['Date'] = df['Date'].apply(pd.to_datetime)
            df.set_index('Date', inplace=True)

            ax = fig.add_subplot(rows, columns, count)
            ax.set_title(symbols[count-1])

            # Get the next color from the cycler
            c = next(colors)["color"]
            df[attr].plot(x='Date', y=attr, grid=True, color=c)

        fig.tight_layout()
        fig.savefig('./src/results/results.pdf')
        plt.show()