""" Entry point for model infering. """
import os

import pandas as pd

from model.classifier import Classifier
from data_utils.preprocessing import encode_sentences, tokenize_sentences

classifier = Classifier("data/best_model/model.meta", "data/best_model/model")

def main():
    classifier = Classifier("data/best_model/model.meta",
                            "data/best_model/model")
    sentences = [["0",  "I used to like cars."]]
    tokenized_sentences = tokenize_sentences(sentences)
    encoded_sentences, _ = encode_sentences(tokenized_sentences, 32)
    result = classifier.infer(encoded_sentences)
    print(result)


def classify(tweet_body):
    sentences = [["0", tweet_body]]
    tokenized_sentences = tokenize_sentences(sentences)
    encoded_sentences, _ = encode_sentences(tokenized_sentences, 32)
    result = classifier.infer(encoded_sentences)
    return result


def _get_sentiment_vector_nasdaq(date, tweets_dir):
    all_data = pd.read_csv('./%s/%s.csv' % (tweets_dir, date))
    tweets_body = all_data['Tweet content']
    sentiment_vector = [classify(row) for row in tweets_body[1:]]
    return sentiment_vector


def _get_sentiment_vector_api(date, tweets_dir):
    # import ipdb; ipdb.set_trace()
    all_data = pd.read_csv('./%s/%s.csv' % (tweets_dir, date))
    # tweets_body = all_data['Tweet content']
    sentiment_vector = [classify(row) for row in all_data[1:]]
    return sentiment_vector


def _get_sentiment_vector_person(date, tweets_dir, person):
    # import ipdb; ipdb.set_trace()
    all_data = pd.read_csv('./%s/%s.csv' % (tweets_dir, date))
    # tweets_body = all_data['Tweet content']
    sentiment_vector = [classify(row) for row in all_data[1:]]
    return sentiment_vector


def write_sentiment_csv_person(tweets_dir, person):
    all_files = os.listdir('./%s' % tweets_dir)
    for file in all_files:
        print('Starting to compute sentiment vector')
        sentiment_vector = _get_sentiment_vector_person(file[:-4], tweets_dir, person)
        print('Computed sentiment vector')
        sentiment_score_f = open('./%s/%s_score.csv' % (tweets_dir, file[:-4]),
                                 'w')
        for elem in sentiment_vector:
            sentiment_score_f.write('%s\n' % elem)


def write_sentiment_csv(tweets_dir):
    all_files = os.listdir('./%s' % tweets_dir)
    for file in all_files:
        print('Starting to compute sentiment vector')
        sentiment_vector = _get_sentiment_vector_api(file[:-4], tweets_dir)
        print('Computed sentiment vector')
        sentiment_score_f = open('./%s/%s_score.csv' % (tweets_dir, file[:-4]),
                                 'w')
        for elem in sentiment_vector:
            sentiment_score_f.write('%s\n' % elem)


if __name__ == "__main__":
    write_sentiment_csv('searched_tweets')
    write_sentiment_csv_person('musk_grouped', 'elonmusk')
