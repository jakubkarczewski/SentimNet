""" Entry point for model training. """


from model.basic_model import BasicModel
from data_utils.preprocessing import CHARSET
from data_utils.dataset import Dataset


def main():
    charset_size = len(CHARSET)
    max_word_len = 32
    build_params = {"batch_size": 50, "max_word_len": max_word_len,
                    "charset_size": charset_size}
    model = BasicModel(
        build_params, "data/logs/run_1/train", "data/logs/run_1/validation",
        "data/logs/run_1/checkpoints", "data/logs/run_1/best_model"
    )
    dataset = Dataset(
        "data/dataset/train_set.csv", "data/dataset/validation_set.csv",
        "data/dataset/test_set.csv", max_word_len
    )
    train_params = {}
    model.train(dataset, **train_params)


if __name__ == "__main__":
    main()