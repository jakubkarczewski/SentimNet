""" Entry point for model infering. """
import os

import pandas as pd

from model.classifier import Classifier
from data_utils.preprocessing import encode_sentences, tokenize_sentences
import split_nasdaq_tweets


def main():
    classifier = Classifier("data/best_model/model.meta",
                            "data/best_model/model")
    sentences = [["0",  "I like vegetables."]]
    tokenized_sentences = tokenize_sentences(sentences)
    encoded_sentences, _ = encode_sentences(tokenized_sentences, 32)
    result = classifier.infer(encoded_sentences)
    print(result)


def classify(tweet_body):
    classifier = Classifier("data/best_model/model.meta",
                            "data/best_model/model")
    sentences = [["0", tweet_body]]
    tokenized_sentences = tokenize_sentences(sentences)
    encoded_sentences, _ = encode_sentences(tokenized_sentences, 32)
    result = classifier.infer(encoded_sentences)
    return result


def _get_sentiment_vector(date, tweets_dir):
    all_data = pd.read_csv('./%s/%s.csv' % (tweets_dir, date))
    tweets_body = all_data['Tweet content']
    sentiment_vector = [classify(row) for row in tweets_body[1:]]
    return sentiment_vector


def write_sentiment_csv(tweets_dir):
    all_files = os.listdir('./%s' % tweets_dir)
    for file in all_files:
        sentiment_vector = _get_sentiment_vector(file[:-4], tweets_dir)
        sentiment_score_f = open('./%s/%s_score.csv' % (tweets_dir, file[:-4]),
                                 'w')
        for elem in sentiment_vector:
            sentiment_score_f.write('%s\n' % elem)


if __name__ == "__main__":
    #main()
    split_nasdaq_tweets.split_datasets(
       'corp_tweets/export_dashboard_aapl_2016_06_15_14_30_09.xlsx',
       1, 'nasdaq_tweets')
    write_sentiment_csv('nasdaq_tweets')
