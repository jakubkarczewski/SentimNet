""" This module contains basic preprocessing functions. """


import csv
from random import shuffle
from os import path


# Output filenames used by split function.
TRAIN_FILENAME = "train_set.csv"
VAL_FILENAME = "validation_set.csv"
TEST_FILENAME = "test_set.csv"

# Indices of values extracted from original dataset lines.
CLASS_I = 0
TEXT_I = -1


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
