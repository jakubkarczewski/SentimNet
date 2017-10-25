""" Entry point for model infering. """


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
    sentences = [["0", tweet_body]]
    tokenized_sentences = tokenize_sentences(sentences)
    encoded_sentences, _ = encode_sentences(tokenized_sentences, 32)
    result = classifier.infer(encoded_sentences)
    return result
    
    
if __name__ == "__main__":
    main()
