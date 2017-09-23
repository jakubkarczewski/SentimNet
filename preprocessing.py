""" This script is an entry point for dataset preprocessing. """


from data_utils.preprocessing import split_dataset


def main():
    split_dataset("data/dataset/training.1600000.processed.noemoticon.csv",
                  "data/dataset/testdata.manual.2009.06.14.csv",
                  "data/dataset")


if __name__ == "__main__":
    main()
