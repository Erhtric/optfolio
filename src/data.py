from typing import List
import pandas as pd
from yahoofinancials import YahooFinancials

def get_history_data(tickers:List[str], start_date:str, end_date:str):
    """The method downloads the history of data from finance.yahoo.com
    of the given symbols.
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
    DATA.to_csv(f'./src/data/ASSET_DATA_{start_date}_to_{end_date}.csv')
    print('Procedure complete!')
