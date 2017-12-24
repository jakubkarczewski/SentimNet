import datetime

import matplotlib.pyplot as plt
import pandas_datareader as web
import numpy as np

import agregate_twitter_scores


def plot_stock_data(start, end, stock, feature):
    prices = web.DataReader(stock, "yahoo", start, end)
    print(len(prices[feature]))
    plt.figure(1, figsize=(15, 10))
    plt.plot(prices[feature])
    plt.ylabel('Dolary ($)')
    plt.xticks(rotation=50)
    plt.title('Cena akcji dla %s' % stock)
    plt.figure(1).savefig('temp/cena_akcji_dla_%s' % stock)


def plot_sentiment(company):
    sentiment_scores = agregate_twitter_scores\
        .iter_through_scores('searched_tweets', company)
    plt.figure(2, figsize=(15, 10))
    dates = [x[1] for x in sentiment_scores]
    values = [x[0] for x in sentiment_scores]
    plt.xlabel("Czas")
    plt.xticks(range(-1, len(values)), dates, rotation=50)
    plt.ylabel('Sentyment')
    plt.plot(values)
    plt.title('Sentyment dla %s' % company)
    plt.figure(3, figsize=(15, 10))
    values_norm = [(value - min(values))/(max(values) - min(values))
                   for value in values]
    plt.xticks(range(-1, len(values)), dates, rotation=50)
    plt.ylabel('Znormalizowany sentyment')
    plt.plot(values_norm)
    plt.title('Znormalizowany sentyment dla %s' % company)
    plt.figure(3).savefig('temp/znormalizowany_sentyment_dla_%s' % company)
    plt.figure(2).savefig('temp/sentyment dla %s' % company)


def plot_stock_and_sentiment(start, end, stock, feature, company):
    # values for plotting
    prices = web.DataReader(stock, "yahoo", start, end)
    sentiment_scores = agregate_twitter_scores \
        .iter_through_scores('searched_tweets', company)
    dates = [x[1] for x in sentiment_scores]
    values = [x[0] for x in sentiment_scores]
    stock_val = prices[feature].values

    fig, ax1 = plt.subplots(figsize=(10, 8))
    sent = ax1.plot(values, 'b-')
    ax1.set_ylabel('Sentyment (jednostki sentymentu)')
    ax1.set_xlabel('Czas (dni)')
    ax1.set_xticks(range(0, len(values)))
    ax1.set_xticklabels(dates, rotation=50)

    ax2 = ax1.twinx()
    price = ax2.plot(stock_val, 'r-')
    ax2.set_ylabel('Cena akcji ($)')

    fig.tight_layout()
    plt.subplots_adjust(top=0.95)
    ax1.legend([sent[0], price[0]], ['Sentyment', 'Cena akcji'])
    plt.title('Cena akcji i sentyment dla %s' % company)
    plt.savefig('temp/cena_i_sent_dla_%s' % company)
    plt.show()


def plot_cross_corr(start, end, stock, feature, company):
    prices = web.DataReader(stock, "yahoo", start, end)
    sentiment_scores = agregate_twitter_scores \
        .iter_through_scores('searched_tweets', company)
    stocks = prices[feature]
    sentiment = [x[0] for x in sentiment_scores]
    cross_corr = np.correlate(stocks, sentiment, mode='full')

    plt.figure(4, figsize=(10, 8))
    sentiment_norm = [(value - min(sentiment))/(max(sentiment) - min(sentiment))
                      for value in sentiment]
    stocks_norm = [(value - min(stocks)) / (max(stocks) - min(stocks))
                   for value in stocks]
    cross_corr_norm = np.correlate(stocks_norm, sentiment_norm, mode='full')
    plt.plot(cross_corr_norm)
    plt.ylabel('Współczynnik korelacji Pearsona')
    plt.xlabel('Opóźnienie sentymentu względem ceny akcji (dni)')
    indexes = np.arange(-(len(sentiment)-1), len(sentiment))
    plt.xticks(range(0, len(indexes)), indexes )
    plt.title('Znormalizowana korelacja krzyżowa dla %s' % company)

    cli_look = np.concatenate((np.arange(-(len(sentiment)-1), len(sentiment))[None,...], cross_corr[None,...]), axis=0)

    plt.figure(4).savefig('temp/korelacja_krzy_dla_%s' % company)
    return cli_look

if __name__ == '__main__':
    # example usage
    start = datetime.datetime(2016, 11, 2)
    end = datetime.datetime(2016, 11, 23)
    feature = 'Adj Close'
    stock = 'FB'
    plot_cross_corr(start, end, stock, feature, 'Facebook')
    plot_stock_and_sentiment(start, end, stock, feature, 'Facebook')


