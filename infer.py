""" Entry point for model infering. """
from nltk.tokenize import sent_tokenize

from model.classifier import Classifier
from data_utils.preprocessing import encode_sentences, tokenize_sentences


def classify_tweet(tweet):
    classifier = Classifier("data/best_model/model.meta",
                            "data/best_model/model")
    parsed_sentences = [["0", tokenize_sentences(sentence)]
                        for sentence in sent_tokenize(tweet)]
    encoded_sentences, _ = encode_sentences(parsed_sentences, 32)
    sentiment_result = classifier.infer(encoded_sentences)

    return tweet, sentiment_result


def main():
    classifier = Classifier("data/best_model/model.meta",
                            "data/best_model/model")
    sentences = [["0",  "I like vegetables."]]
    tokenized_sentences = tokenize_sentences(sentences)
    encoded_sentences, _ = encode_sentences(tokenized_sentences, 32)
    result = classifier.infer(encoded_sentences)
    print(result)


if __name__ == "__main__":
    main()
