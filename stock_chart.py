import datetime

import matplotlib.pyplot as plt
import pandas_datareader as web
import numpy as np

import agregate_twitter_scores


def plot_stock_data(start, end, stock, feature):
    prices = web.DataReader(stock, "yahoo", start, end)
    print(len(prices[feature]))
    plt.figure(1)
    plt.plot(prices[feature])
    plt.ylabel('Dollars ($)')
    plt.xticks(rotation=50)
    plt.title('Stock price for %s' % stock)


def plot_sentiment(company):
    sentiment_scores = agregate_twitter_scores\
        .iter_through_scores('searched_tweets', company)
    plt.figure(2)
    dates = [x[1] for x in sentiment_scores]
    values = [x[0] for x in sentiment_scores]
    plt.xticks(range(-1, len(values)), dates, rotation=50)
    plt.ylabel('Sentiment score')
    plt.plot(values)
    plt.title('Sentiment score for %s' % company)
    plt.figure(3)
    values_norm = [(value - min(values))/(max(values) - min(values))
                   for value in values]
    plt.xticks(range(-1, len(values)), dates, rotation=50)
    plt.ylabel('Sentiment score normalized')
    plt.plot(values_norm)
    plt.title('Normalized sentiment score for %s' % company)


def plot_cross_corr(start, end, stock, feature):
    prices = web.DataReader(stock, "yahoo", start, end)
    sentiment_scores = agregate_twitter_scores \
        .iter_through_scores('searched_tweets', 'Tesla')
    stocks = prices[feature]
    sentiment = [x[0] for x in sentiment_scores]
    cross_corr = np.correlate(stocks, sentiment, mode='full')
    plt.figure(4)
    plt.plot(cross_corr)
    plt.ylabel('Correlation coef.')
    plt.xlabel('Lag of sentiment score in relation to stock price')
    indexes = np.arange(-(len(sentiment)-1), len(sentiment))
    plt.xticks(range(0, len(indexes)), indexes )
    plt.title('Cross correlation for %s' % stock)

    plt.figure(5)
    sentiment_norm = [(value - min(sentiment))/(max(sentiment) - min(sentiment))
                      for value in sentiment]
    stocks_norm = [(value - min(stocks)) / (max(stocks) - min(stocks))
                   for value in stocks]
    cross_corr_norm = np.correlate(stocks_norm, sentiment_norm, mode='full')
    plt.plot(cross_corr_norm)
    plt.ylabel('Correlation coef.')
    plt.xlabel('Lag of sentiment score in relation to stock price')
    indexes = np.arange(-(len(sentiment)-1), len(sentiment))
    plt.xticks(range(0, len(indexes)), indexes )
    plt.title('Normalized cross correlation for %s' % stock)

    cli_look = np.concatenate((np.arange(-(len(sentiment)-1), len(sentiment))[None,...], cross_corr[None,...]), axis=0)

    return cli_look

if __name__ == '__main__':
    # example usage
    start = datetime.datetime(2016, 11, 2)
    end = datetime.datetime(2016, 11, 23)
    feature = 'Adj Close'
    stock = 'FB'
    plot_stock_data(start, end, stock, feature)
    plot_sentiment('Facebook')
    print(plot_cross_corr(start, end, stock, feature))
    plt.show()



