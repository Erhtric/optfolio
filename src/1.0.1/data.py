from typing import List
import pandas as pd
import numpy as np
from yahoofinancials import YahooFinancials

def get_history_data(tickers:List[str], start_date:str, end_date:str):
    """The method downloads the history of data from finance.yahoo.com
    of the given symbols.

    Args:
        tickers (List[str]): the objective ticker symbols
        start_date (str): history data's starting date
        end_date (str): history data's ending date
    """
    print('...Downloading data...')
    print(f'Starting date: {start_date}')
    print(f'Ending date: {end_date}')

    frames = []
    for ticker in tickers:
        tickerData = YahooFinancials(ticker).get_historical_price_data(start_date,
                                                                        end_date,
                                                                        time_interval='daily')
        tickerDf = pd.DataFrame(tickerData[ticker]['prices']).drop('date', axis=1).set_index('formatted_date')
        tickerDf.insert(0, 'ticker', ticker, True)

        # tickerDf.to_csv(f'./src/1.0.1/data/{ticker}.csv')
        frames.append(tickerDf)

    print('...saving data to file...')
    DATA = pd.concat(frames)
    DATA.to_csv('./src/1.0.1/data/ASSET_DATA.csv')
    print('Procedure complete!')
