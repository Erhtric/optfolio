from typing import List
import numpy as np
from matplotlib import pyplot as plt
import yfinance as yf
import pandas as pd
class DataHandler:

    tickerSymbols = []
    start_date = ''
    end_date = ''

    def __init__(self, tickers: List[str]):
        self.tickerSymbols = tickers

    def set_tickerSymbols(self, tickers):
        """It adds tickers to the list of active ticker symbols."""
        self.tickerSymbols += tickers

    def remove_tickerSymbol(self, ticker: str):
        """It removes the ticker specified from the list of active ticker symbols."""
        self.tickerSymbols.remove(ticker)

    def get_history_data(self, start_time: str, end_time: str, period: str, save_file: bool=True) -> pd.DataFrame:
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
        self.start_date = start_time
        self.end_date = end_time

        frames = []
        for ticker in self.tickerSymbols:
            tickerData = yf.Ticker(ticker)
            tickerDf = tickerData.history(start=start_time, end=end_time, period=period)
            tickerDf['Ticker'] = ticker
            if save_file: tickerDf.to_csv('./src/data/{}.csv'.format(ticker))
            frames.append(tickerDf)

        df = pd.concat(frames)
        return df

    def get_info_data(self, keyword: str) -> List:
        """The method get the relative info dictated with the relative keyword

        Args:
            symbols: the objective symbols
            keyword: the keyword used to get the information

        Returns:
            List: the list of values extracted
        """
        values = []
        for ticker in self.tickerSymbols:
            dict_info = yf.Ticker(ticker).info
            values.append(dict_info[keyword])

        return values

    def __str__(self) -> str:
        return f'Active ticker symbols: {self.tickerSymbols}'

    def plot_time_series(self, attr: str):
        """The method plots the time series for each ticker in the already selected
        chosen date interval.

        Args:
            attr: the attribute of the Dataframe to plot, it is possbile to choose one of those:
                Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
        """
        print("...Plotting \"{}\" for the following tickers: {}...".format(attr, self.tickerSymbols))

        n = len(self.tickerSymbols)
        columns = 2 if n >= 2 else n
        rows = int(np.ceil(n / 2))

        fig = plt.figure()
        fig.suptitle(attr + ' prices')
        for count in range(1, n+1):
            df = pd.read_csv('./src/data/{}.csv'.format(self.tickerSymbols[count-1]))
            # Convert the Date column to the datetime type.
            df['Date'] = df['Date'].apply(pd.to_datetime)
            df.set_index('Date', inplace=True)

            ax = fig.add_subplot(rows, columns, count)
            ax.set_title(self.tickerSymbols[count-1])
            df[attr].plot(grid=True)

            fig.tight_layout()
            fig.savefig('./src/results/results.pdf')