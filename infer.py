""" Entry point for model infering. """
import pandas as pd

from model.classifier import Classifier
from data_utils.preprocessing import encode_sentences, tokenize_sentences


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
    all_data = pd.read_csv('%s/%s.csv' % (tweets_dir, date))
    tweets_body = all_data['Tweet content']
    sentiment_vector = [classify(row) for row in tweets_body[1:]]
    return sentiment_vector

    
if __name__ == "__main__":
    main()
    # print(_get_sentiment_vector('2016-05-03', 'nasdaq_tweets'))
