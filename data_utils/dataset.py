""" This module contains dataset object definition. """


import string
from random import shuffle

import numpy as np
from nltk.tokenize import word_tokenize

from data_utils.preprocessing import read_dataset, CLASS_I, TEXT_I


class Dataset:
    """
    This class wraps train, validation and test sets and provides methods, which
    allow supply of data for training.
    """

    # Constants used in encoding.
    CHARSET = string.ascii_lowercase + string.punctuation + string.digits
    ENCODER = {char: i for i, char in enumerate(CHARSET)}

    # Constant indices of particular datasets in _datasets list.
    TRAIN_I = 0
    VAL_I = 1
    TEST_I = 2

    def __init__(self, train_file_path, val_file_path, test_file_path,
                 max_word_len):
        self._datasets = []
        self._minibatches_is = [0, 0, 0]
        self._max_word_len = max_word_len
        for file_path in [train_file_path, val_file_path, test_file_path]:
            self._datasets.append(read_dataset(file_path, trim_columns=False))

    def get_next_minibatch(self, set_i, minibatch_size):
        """ Retrieve next minibatch from the target set. """
        # If there are not enough sentences left, then reshuffle the dataset.
        if (
            self._minibatches_is[set_i] + minibatch_size
            >= len(self._datasets[set_i])
        ):
            self._minibatches_is[set_i] = 0
            shuffle(self._datasets[set_i])
        sentences = (self._datasets[set_i]
                     [self._minibatches_is[set_i]:
                      self._minibatches_is[set_i] + minibatch_size])
        for sen in sentences:
            print(sen)
        return self._encode_sentences(sentences)

    def _encode_sentence(self, sentence):
        """
        Encode sentence as one-hot tensor of shape [None, MAX_WORD_LENGTH,
        CHARSET_SIZE].
        """
        sentence = sentence.lower()
        encoded_sentence = []
        words = word_tokenize(sentence)
        print(words)
        sentence_len = len(words)
        for word in words:
            # Encode every word as matrix of shape [MAX_WORD_LENGTH,
            # CHARSET_SIZE] where each valid character gets encoded as one-hot
            # row vector of word matrix.
            encoded_word = np.zeros([self._max_word_len, len(Dataset.CHARSET)])
            for char, encoded_char in zip(word, encoded_word):
                if char in Dataset.CHARSET:
                    encoded_char[Dataset.ENCODER[char]] = 1.0
            encoded_sentence.append(encoded_word)
        return np.array(encoded_sentence), sentence_len

    def _encode_sentences(self, sentences):
        """ Encode group of sentences into one big tensor. """
        encoded_sentences = []
        labels = []
        max_len = 0
        for sentence in sentences:
            # Encode class (negative/positive) into one-hot vector.
            labels.append([1, 0] if sentence[CLASS_I] == "0" else [0, 1])
            encoded_sentence, sentence_len = self._encode_sentence(
                sentence[TEXT_I]
            )
            encoded_sentences.append(encoded_sentence)
            if sentence_len > max_len:
                max_len = sentence_len
        expanded_sentences = self._expand_sentences(encoded_sentences)
        return expanded_sentences, np.array(labels)

    def _expand_sentences(self, encoded_sentences):
        """
        Expand every given sentence to the same length (pad it with 0) and
        put them all into one tensor.
        """
        # Get word counts of all sentences.
        lens = np.array([sentence.shape[0] for sentence in encoded_sentences])
        # Create mask of shape [num_of_sentences, max_sentence_len] in which
        # each row contains information about where to put words into expanded
        # sentence tensor. For example for sentence of length 4 in a set in
        # which the longest sentence had 12 words this row would look like this
        # [True, True, True, True, False, ...] and it would mean that we should
        # put out four words at the start of the sentence and leave the rest
        # unchanged (set to 0.0).
        insertion_mask = np.arange(lens.max()) < lens[:, np.newaxis]
        expanded_sentences = np.zeros(
            [len(encoded_sentences), lens.max(), self._max_word_len,
             len(Dataset.CHARSET)]
        )
        expanded_sentences[insertion_mask] = np.concatenate(encoded_sentences)
        return expanded_sentences
