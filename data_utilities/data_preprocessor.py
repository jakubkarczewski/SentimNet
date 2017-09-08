
import numpy as np
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import string
import random, csv, os

# Paths to dataset files
TRAIN_SRC = 'data/training.1600000.processed.noemoticon.csv'
TEST_SRC = 'data/testdata.manual.2009.06.14.csv'
TEST_SET = 'data/test_set.csv'
VALID_SET = 'data/valid_set.csv'
TRAIN_SET = 'data/train_set.csv'
PROJECT_PATH = '/home/kuba/Development/SentNet/'
# Percentage of training set we wish to sacrifice for validation
VALIDATION_PERCENTAGE = 0.03


def merge_lines(lines):
    # Function with which we avoid redundant splitting to comma-separated chars
    # and "," separated words
    new_lines = []
    for l in lines:
        splited = l.split('","')
        new_lines.append((splited[0][1:], splited[-1][:-2]))
    return new_lines

def prepare_valid_datasets():
    # check if paths are valid
    assert os.path.exists(PROJECT_PATH+TRAIN_SRC)
    assert os.path.exists(PROJECT_PATH + TEST_SRC)

    # create validation set from training data
    print 'Building validation and training sets.'

    with open(PROJECT_PATH + TRAIN_SRC, 'r') as training_src:
        lines = training_src.readlines()
        random.shuffle(lines)
        # point a dividing indices and divide using colon notation
        threshold = int((1 - VALIDATION_PERCENTAGE)*len(lines))
        validation_lines = lines[threshold:]
        training_lines = lines[threshold:]

    with open(PROJECT_PATH + TRAIN_SET, 'w') as validation_file:
        writer = csv.writer(validation_file)
        writer.writerows(merge_lines(validation_lines))

    with open(PROJECT_PATH + VALID_SET, 'w') as training_file:
        writer = csv.writer(training_file)
        writer.writerows(merge_lines(training_lines))

    # create testing set
    print 'Building testing set.'

    with open(PROJECT_PATH + TEST_SRC, 'r') as testing_src:
        lines = testing_src.readlines()
        random.shuffle(lines)

    with open(PROJECT_PATH + TEST_SET, 'w') as testing_file:
        writer = csv.writer(testing_file)
        writer.writerows(merge_lines(lines))

    print 'Good job! Datasets have been created!'

if __name__ == '__main__':
    prepare_valid_datasets()






