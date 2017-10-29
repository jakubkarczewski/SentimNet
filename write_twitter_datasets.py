import csv
import subprocess
from datetime import date
from datetime import timedelta
import tweepy


def _configure_connection(dir_name):
    ckey = "uwQoag4aBHKJcFsOsKi8q94iS"
    csecret = "rz89scxAxLla6OQDmNUyQ0TFCZt5L4IT6XoVzBEfRmcuRTgvS2"
    atoken = "878227501880946688-zaypdJupSY5bQOYg6DIbGTwIp3JlF50"
    asecret = "hafrjgx2zoIVATBLmb4HppsaDiQLLkZGZjqfYXK6D6qSc"

    OAUTH_KEYS = {'consumer_key': ckey,
                  'consumer_secret': csecret,
                  'access_token_key': atoken,
                  'access_token_secret': asecret}
    auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'],
                               OAUTH_KEYS['consumer_secret'])
    api = tweepy.API(auth)
    subprocess.call(["mkdir", dir_name])
    return api


def collect_tweets(dataset_filters, dir_name, limit=0):
    """
    Takes as input a tuple of 3-element sub-tuples in form of:
    ('tag', 'start_date', 'end_date'). Tag is a phrase with which we will
    filter the tweets. Date constrain the tweets in terms of time of posting.
    Note that we are restrained to 7 days retention time.
    The method writes n datasets files containing comma separated tweets
    (in .csv format and n = len(filters)).
    """
    api = _configure_connection(dir_name)
    raw_data = []

    for dataset_filter in dataset_filters:
        collected_tweets = tweepy.Cursor(api.search,
                                         since=dataset_filter[1],
                                         until=dataset_filter[2],
                                         q=dataset_filter[0],
                                         lang="en").items(limit)
        parsed_tweets = [(tweet.retweet_count, tweet.text) for tweet
                         in collected_tweets]
        raw_data.append(parsed_tweets)

    for raw_dataset, dataset_filter in zip(raw_data, dataset_filters):
        with open('%s/%s_from_%s.csv' % (dir_name,
                                         dataset_filter[0],
                                         dataset_filter[1]), 'w') as csvfile:
            dataset_writer = csv.writer(csvfile, delimiter=',')
            dataset_writer.writerow(raw_dataset)

    return True


def collect_tweets_weekly(start_date, end_date, stock_filters, dir_name,
                          limit=20):
    """
    Start and end date need to be in form: ie. date(2017, 8, 8)
    Stock filters needs to be a tuple of strings containing tags for filtering.
    """
    d1 = start_date
    d2 = end_date
    delta = d2 - d1
    all_days = [str(start_date + timedelta(days=i)) for i
                in range(delta.days + 1)]
    args_list = []
    for filter in stock_filters:
        for i in range(len(all_days)-1):
            args_list.append((filter, all_days[i], all_days[i+1]))

    collect_tweets(tuple(args_list), dir_name, limit=limit)
    return True


if __name__ == '__main__':
    if(collect_tweets_weekly(date(2017, 10, 21), date(2017, 10, 28),
                             ('@tesla', '@google'), 'searched_tweets'), 30):
        print("Datasets ready.")
    else:
        print("Something went wrong.")
