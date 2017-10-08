import csv
import subprocess

import tweepy


def _configure_connection():
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
    subprocess.call(["mkdir", "tweets"])
    return api


def collect_tweets(dataset_filters, limit=0):
    """
    Takes as input a tuple of 3-element sub-tuples in form of:
    ('tag', 'start_date', 'finish_date'). Tag is a phrase with which we will
    filter the tweets. Dates constrain the tweets in terms of time of posting.
    Note that we are restrained to 7 days retention time.
    The method writes n datasets files containing comma separated tweets
    (in .csv format and n = len(filters)).
    """
    api = _configure_connection()
    raw_data = []

    for dataset_filter in dataset_filters:
        collected_tweets = tweepy.Cursor(api.search,
                                         since=dataset_filter[1],
                                         until=dataset_filter[2],
                                         q=dataset_filter[0],
                                         lang="en").items(limit)
        parsed_tweets = [tweet.text for tweet in collected_tweets]
        raw_data.append(parsed_tweets)

    for raw_dataset, dataset_filter in zip(raw_data, dataset_filters):
        with open('tweets/%s_from_%s_to_%s' % (dataset_filter[0],
                                        dataset_filter[1],
                                        dataset_filter[2]), 'w') as csvfile:
            dataset_writer = csv.writer(csvfile, delimiter=',')
            dataset_writer.writerow(raw_dataset)

    return True

if __name__ == '__main__':
    args = (('@tesla', '2017-10-06', '2017-10-08'),
            ('@google', '2017-10-06', '2017-10-08'))
    if collect_tweets(args, limit=20):
        print("Datasets ready.")
    else:
        print("Something went wrong.")
