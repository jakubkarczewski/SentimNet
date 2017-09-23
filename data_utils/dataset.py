""" This module contains dataset object definition. """


from random import shuffle

from data_utils.preprocessing import *


class Dataset:
    """
    This class wraps train, validation and test sets and provides methods, which
    allow supply of data for training.
    """

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
            sentences = read_dataset(file_path, trim_columns=False)
            tokenized_sentences = tokenize_sentences(sentences)
            self._datasets.append(tokenized_sentences)

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
        self._minibatches_is[set_i] += minibatch_size
        return encode_sentences(sentences, self._max_word_len)

    def get_set_size(self, set_i):
        """ Return size of a selected dataset. """
        return len(self._datasets[set_i])
