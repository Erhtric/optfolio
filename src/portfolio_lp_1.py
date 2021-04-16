import data_handler
import numpy as np
from matplotlib import pyplot as plt

data = data_handler.DataHandler(["GME", "AMC", "AAPL", "MSFT"])
#data.get_history_data

#print(data.get_history_data(start_time="2020-01-01", end_time="2020-03-01").to_numpy.shape)
#data.get_history_data(start_time="2010-01-01", end_time="2020-01-01", period='7d')


data.plot_time_series('Close')