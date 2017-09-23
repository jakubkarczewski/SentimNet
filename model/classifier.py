""" This module contains class capable of infering with trained model. """


import tensorflow as tf


class Classifier:
    """
    A class for inferring the class without initializing those graph nodes
    that are needed only for training.
    """

    # Names used to load necessary nodes from trained model file.
    SCORES_NAME = "scores"
    IN_SENTENCES_NAME = "in_sentences"
    IN_LSTM_DROPOUT_NAME = "in_lstm_dropout"

    def __init__(self, best_meta_file_path, best_model_file_path):
        """ Setup environment. """
        meta_file_path = best_meta_file_path
        model_file_path = best_model_file_path
        self._session = tf.Session()
        saver = tf.train.import_meta_graph(meta_file_path)
        saver.restore(self._session, model_file_path)
        self._in_sentences = tf.get_collection(self.IN_SENTENCES_NAME)[0]
        self._in_lstm_dropout = tf.get_collection(self.IN_LSTM_DROPOUT_NAME)[0]
        self._scores = tf.get_collection(self.SCORES_NAME)[0]

    def infer(self, in_sentences):
        """ Feed froward x through the loaded net. """
        return self._session.run(
            self._scores,
            feed_dict={self._in_sentences: in_sentences,
                       self._in_lstm_dropout: 0.0}
        )

    def close(self):
        self._session.close()
