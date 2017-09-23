""" This module contains basic preprocessing functions. """


import csv
from random import shuffle
from os import path
import string

import numpy as np
from nltk.tokenize import word_tokenize


__all__ = ["TRAIN_FILENAME", "VAL_FILENAME", "TEST_FILENAME", "CLASS_I",
           "TEXT_I", "validate_paths", "read_dataset", "write_dataset",
           "split_dataset", "tokenize_sentences", "encode_sentence",
           "encode_sentences", "expand_sentences"]


# Output filenames used by split function.
TRAIN_FILENAME = "train_set.csv"
VAL_FILENAME = "validation_set.csv"
TEST_FILENAME = "test_set.csv"

# Indices of values extracted from original dataset lines.
CLASS_I = 0
TEXT_I = -1

# Constants used in encoding.
CHARSET = string.ascii_lowercase + string.punctuation + string.digits
ENCODER = {char: i for i, char in enumerate(CHARSET)}

# Number of detected classes.
NUM_OF_CLASSES = 2


def validate_paths(train_file_path, test_file_path, output_dir):
    """ Check if all paths required for preprocessing exist. """
    if (
        not (path.isfile(train_file_path)
             and path.isfile(test_file_path)
             and path.isdir(output_dir))
    ):
        raise ValueError("Paths provided to preprocessing are not valid.")


def read_dataset(set_file_path, trim_columns):
    """ Read and return class and tweet text from every line of the dataset. """
    lines = []
    with open(set_file_path, "r", encoding="iso-8859-1") as set_file:
        set_csv_reader = csv.reader(set_file)
        for line in set_csv_reader:
            if trim_columns:
                lines.append((line[CLASS_I], line[TEXT_I]))
            else:
                lines.append(line)
    shuffle(lines)
    return lines


def write_dataset(set_file_path, lines):
    """ Output lines to dataset file on the given path. """
    with open(set_file_path, "w", encoding="iso-8859-1", newline="") \
            as set_file:
        writer = csv.writer(set_file)
        writer.writerows(lines)


def split_dataset(train_file_path, test_file_path, output_dir,
                  val_set_size=0.1):
    """
    Load original train and test sets from dataset directory, split train set
    for two parts (train and validation) and save results in the output
    directory.
    """
    validate_paths(train_file_path, test_file_path, output_dir)
    # Read original dataset lines.
    train_lines = read_dataset(train_file_path, trim_columns=True)
    test_lines = read_dataset(test_file_path, trim_columns=True)
    # Split train lines into two parts.
    split_point = int(val_set_size * len(train_lines))
    val_lines = train_lines[:split_point]
    new_train_lines = train_lines[split_point:]
    # Output new datasets to given directory.
    write_dataset(path.join(output_dir, TRAIN_FILENAME), new_train_lines)
    write_dataset(path.join(output_dir, VAL_FILENAME), val_lines)
    write_dataset(path.join(output_dir, TEST_FILENAME), test_lines)


def tokenize_sentences(sentences):
    """ Tokenize all sentences texts beforehand to save computing time. """
    new_sentences = []
    for sentence in sentences:
        new_sentences.append(
            [sentence[CLASS_I], word_tokenize(sentence[TEXT_I].lower())]
        )
    return new_sentences


def encode_sentence(tokenized_sentence, max_word_len):
    """
    Encode sentence as one-hot tensor of shape [None, MAX_WORD_LENGTH,
    CHARSET_SIZE].
    """
    encoded_sentence = []
    sentence_len = len(tokenized_sentence)
    for word in tokenized_sentence:
        # Encode every word as matrix of shape [MAX_WORD_LENGTH,
        # CHARSET_SIZE] where each valid character gets encoded as one-hot
        # row vector of word matrix.
        encoded_word = np.zeros([max_word_len, len(CHARSET)])
        for char, encoded_char in zip(word, encoded_word):
            if char in CHARSET:
                encoded_char[ENCODER[char]] = 1.0
        encoded_sentence.append(encoded_word)
    return np.array(encoded_sentence), sentence_len


def encode_sentences(sentences, max_word_len):
    """ Encode group of sentences into one big tensor. """
    encoded_sentences = []
    labels = []
    max_len = 0
    for sentence in sentences:
        # Encode class (negative/positive) into one-hot vector.
        labels.append([1, 0] if sentence[CLASS_I] == "0" else [0, 1])
        encoded_sentence, sentence_len = encode_sentence(sentence[TEXT_I],
                                                         max_word_len)
        encoded_sentences.append(encoded_sentence)
        if sentence_len > max_len:
            max_len = sentence_len
    expanded_sentences = expand_sentences(encoded_sentences, max_word_len)
    return expanded_sentences, np.array(labels)


def expand_sentences(encoded_sentences, max_word_len):
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
        [len(encoded_sentences), lens.max(), max_word_len, len(CHARSET)]
    )
    expanded_sentences[insertion_mask] = np.concatenate(encoded_sentences)
    return expanded_sentences
