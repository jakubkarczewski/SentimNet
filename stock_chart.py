import datetime

import matplotlib.pyplot as plt
import pandas_datareader as web


def plot_stock_data(start, end, stock, features):
    stock = web.DataReader(stock, "yahoo", start, end)
    for feature in features:
        plt.plot(stock[feature])
    plt.ylabel('Dollars ($)')
    plt.show()

if __name__ == '__main__':
    # example usage
    start = datetime.datetime(2016, 3, 28)
    end = datetime.datetime(2016, 5, 12)
    #features = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    features = ['Adj Close']
    stock = 'NVDA'
    plot_stock_data(start, end, stock, features)



