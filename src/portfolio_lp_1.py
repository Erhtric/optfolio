import data
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

symbols = ["GME", "AAPL"]
df = data.get_history_data(symbols, start_time="2019-01-01", end_time="2020-01-01", period='1d')
#data.plot_time_series(symbols, 'Close')
