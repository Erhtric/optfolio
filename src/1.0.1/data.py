from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# def get_api_key():
#     with open("src/1.0.1/api.config") as file:
#         key = file.readline()
#         return key

# API: xz91aUn4uFjdGn7RL2B4
# def get_closing_prices(tickers):

#     quandl.ApiConfig.api_key = get_api_key()
#     assets = tickers
#     data = quandl.get_table(
#         'WIKI/PRICES',
#         ticker = assets,
#         qopts = {'columns': ['date', 'ticker', 'adj_close']},
#         date = {'gte': '2014-1-1', 'lte': '2016-12-31' }, paginate=True)

    # # reorganise data pulled by setting date as index with
    # # columns of tickers and their corresponding adjusted prices
    # df = data.set_index('date').pivot(columns='ticker')
    # print(df.head)

def get_history_data(tickers:List[str], start_date:str, end_date:str):
    print('...Downloading data...')
    print(f'Starting date: {start_date}')
    print(f'Ending date: {end_date}')

assets = ['CNP', 'F', 'WMT', 'GE', 'TSLA']
sd = '2014-01-01'
ed = '2016-12-31'
# get_closing_prices(assets)
get_history_data(assets, start_date=sd, end_date=ed)