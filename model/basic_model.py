""" This module contains basic model definition. """


import tensorflow as tf
import numpy as np

from model.net_builiding import *
from model.classifier import Classifier
from data_utils.preprocessing import NUM_OF_CLASSES
from data_utils.dataset import Dataset


class BasicModel:
    """ Basic model described in the sentiment analysis tutorial. """

    def __init__(self, build_params, train_root, val_root, checkpoints_root,
                 best_model_root):
        # Temporary paths handling.
        self._train_root = train_root
        self._val_root = val_root
        self._checkpoints_root = checkpoints_root
        self._best_model_root = best_model_root
        self._checkpoint_file_path = self._checkpoints_root + "/model"
        self._best_model_file_path = self._best_model_root + "/model"
        self._create_placeholders(build_params["max_word_len"],
                                  build_params["charset_size"])
        self._build_net(**build_params)
        self._build_training_nodes()
        self._create_summary()
        self._create_environment()
        self._register_infer_nodes()

    def train(
            self, dataset, learning_rate=0.001, desired_loss=0.001,
            max_iterations=1000000, decay_interval=10, decay_rate=1.0,
            save_interval=1000, best_save_interval=200,
            validation_interval=200, lstm_dropout=0.0, batch_size=50
    ):
        """ Public entry method for model's training. """
        self._session.run(tf.global_variables_initializer())
        min_loss = -np.log(1 / NUM_OF_CLASSES)
        for iteration in range(max_iterations):
            train_loss_out, _ = self._do_single_run(
                iteration, "train", dataset, batch_size, learning_rate,
                lstm_dropout, loops_count=10
            )
            if iteration % validation_interval == 0:
                val_loss_out, _ = self._do_single_run(
                    iteration, "validation", dataset, batch_size, 0.0, 0.0
                )
            if iteration % decay_interval == 0:
                learning_rate *= decay_rate
            if iteration % save_interval == 0:
                self._checkpoints_saver.save(
                    self._session, self._checkpoint_file_path,
                    global_step=iteration
                )
            # Save best model basing on validation loss.
            if iteration % best_save_interval == 0 and val_loss_out < min_loss:
                min_loss = val_loss_out
                self._best_model_saver.save(self._session,
                                            self._best_model_file_path)
            if train_loss_out < desired_loss:
                break

    def _do_single_run(self, iteration, run_type, dataset, batch_size,
                       learning_rate, lstm_dropout, loops_count=None):
        """ Perform single run of selected type. """
        losses = []
        accuracies = []
        # Determine which nodes to run and set index.
        nodes_to_run = [self._loss, self._accuracy]
        if run_type == "train":
            nodes_to_run += [self._train]
            set_i = Dataset.TRAIN_I
        elif run_type == "validation":
            set_i = Dataset.VAL_I
        else:
            set_i = Dataset.TEST_I
        # Loop over whole dataset if not specified otherwise.
        if loops_count is None:
            loops_count = (dataset.get_set_size(set_i) // batch_size)
        for i in range(loops_count):
            data, labels = dataset.get_next_minibatch(set_i, batch_size)
            output = self._session.run(
                nodes_to_run,
                feed_dict={self._in_sentences: data, self._in_labels: labels,
                           self._in_lstm_dropout: lstm_dropout,
                           self._in_learning_rate: learning_rate}
            )
            if run_type == "train":
                loss_out, accuracy_out, _ = output
            else:
                loss_out, accuracy_out = output
            losses.append(loss_out)
            accuracies.append(accuracy_out)
        # Log obtained loss and accuracy values.
        mean_loss = np.mean(losses)
        mean_accuracy = np.mean(accuracies)
        self._output_log_to_console(run_type, iteration, mean_loss,
                                    mean_accuracy, learning_rate)
        summary_out, = self._session.run(
            [self._summary],
            feed_dict={self._in_loss: mean_loss,
                       self._in_accuracy: mean_accuracy}
        )
        if run_type == "train":
            self._train_writer.add_summary(summary_out, global_step=iteration)
        elif run_type == "validation":
            self._val_writer.add_summary(summary_out, global_step=iteration)
        return mean_loss, mean_accuracy

    def _create_placeholders(self, max_word_len, charset_size):
        """ Create necessary model's placeholders. """
        # Sentence length varies depending on the current minibatch.
        self._in_sentences = tf.placeholder(
            tf.float32, shape=[None, None, max_word_len, charset_size]
        )
        self._in_labels = tf.placeholder(tf.float32,
                                         shape=[None, NUM_OF_CLASSES])
        self._in_learning_rate = tf.placeholder(tf.float32, shape=[])
        self._in_lstm_dropout = tf.placeholder(tf.float32, shape=[])

    def _build_net(self, kernels_widths=(1, 2, 3, 4, 5, 6, 7),
                   filters_counts=(25, 50, 75, 100, 125, 150, 175),
                   max_word_len=32, charset_size=68, highway_layers_count=1,
                   lstm_layers_sizes=(512, )):
        """ Build whole network. """
        cnn_out = build_char_embedding(self._in_sentences, kernels_widths,
                                       filters_counts, max_word_len,
                                       charset_size)
        cnn_out_size = np.sum(filters_counts)
        highway_out = build_highway_net(cnn_out, cnn_out_size,
                                        highway_layers_count)
        # Split words back into sentences for lstm.
        lstm_in = tf.reshape(
            highway_out, [tf.shape(self._in_sentences)[0], -1, cnn_out_size]
        )
        lstm_cell = build_lstm_cell(lstm_layers_sizes, self._in_lstm_dropout)
        outputs, final_lstm_state = tf.nn.dynamic_rnn(lstm_cell, lstm_in,
                                                      dtype=tf.float32)
        # Extract only last lstm output.
        lstm_out = outputs[:, -1]
        self._output = build_linear_layer("output", lstm_out,
                                          lstm_out.shape[-1], NUM_OF_CLASSES)
        self._scores = tf.nn.softmax(self._output)
        self._predictions = tf.argmax(self._output, axis=1)

    def _build_training_nodes(self):
        """ Create training nodes. """
        self._loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self._output,
                                                    labels=self._in_labels)
        )
        self._accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self._predictions, tf.argmax(self._in_labels, 1)),
                    dtype=tf.float32)
        )
        self._train = tf.train.AdamOptimizer(
            learning_rate=self._in_learning_rate
        ).minimize(self._loss)

    def _create_summary(self):
        """ Create train and validation summary nodes. """
        self._in_loss = tf.placeholder(tf.float32, [])
        self._in_accuracy = tf.placeholder(tf.float32, [])
        tf.summary.scalar("loss", self._in_loss)
        tf.summary.scalar("accuracy", self._in_accuracy)
        self._summary = tf.summary.merge_all()

    def _create_environment(self):
        """ Create training environment. """
        self._session = tf.Session()
        self._best_model_saver = tf.train.Saver(max_to_keep=1)
        self._checkpoints_saver = tf.train.Saver(max_to_keep=10)
        self._train_writer = tf.summary.FileWriter(self._train_root)
        self._val_writer = tf.summary.FileWriter(self._val_root)

    def _register_infer_nodes(self):
        """ Register graph nodes used in classification with trained model. """
        tf.add_to_collection(Classifier.IN_SENTENCES_NAME, self._in_sentences)
        tf.add_to_collection(Classifier.IN_LSTM_DROPOUT_NAME,
                             self._in_lstm_dropout)
        tf.add_to_collection(Classifier.SCORES_NAME, self._scores)

    @staticmethod
    def _output_log_to_console(run_type, iteration, loss, accuracy,
                               learning_rate):
        """ Print training log to stdout. """
        print(
            "{} - Iteration {}:\n\tLoss: {}\n\tAccuracy: {}""\n\tLearning "
            "rate: {}".format(run_type, iteration, loss, accuracy,
                              learning_rate)
        )
