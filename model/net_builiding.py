""" This module provides tensorflow graph building functions. """


import tensorflow as tf


__all__ = ["build_char_embedding", "build_highway_layer", "build_highway_net",
           "build_linear_layer", "build_lstm_cell"]


def build_char_embedding(in_sentences, kernels_widths, filters_counts,
                         max_word_len, charset_size):
    """ Create convolution character embedding layer. """
    # Reshape input sentences for convolution purposes to tensor in which every
    # word becomes "image" with 1 height, max_word_len width and as many
    # channels as there are characters in our charset. We have batch_size *
    # max_sentence_len of these word "images" and we need to do it this way,
    # because char_cnn is supposed to look at every word separately to embed it.
    in_sentences = tf.reshape(in_sentences, [-1, 1, max_word_len, charset_size])
    cnn_outs = []
    with tf.variable_scope("char_cnn"):
        for kernel_width, filters_count in zip(kernels_widths, filters_counts):
            out_word_width = max_word_len - kernel_width + 1
            conv_out = tf.layers.conv2d(
                inputs=in_sentences, filters=filters_count,
                kernel_size=[1, kernel_width], padding="valid",
                activation=tf.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
            )
            # After convolution in_sentences transform to tensor of shape
            # [batch_size * max_sentence_len, 1, out_word_width, filters_count].
            pool_out = tf.nn.max_pool(conv_out, [1, 1, out_word_width, 1],
                                      strides=[1, 1, 1, 1], padding="VALID")
            # After max pooling we have tensor of shape [batch_size *
            # max_sentence_len, 1, 1, filters_count], so we squeeze unnecessary
            # dimensions to obtain 1D character encoding of each word.
            cnn_outs.append(tf.squeeze(pool_out, [1, 2]))
    output = tf.concat(cnn_outs, 1)
    return output


def build_linear_layer(name, in_data, in_size, out_size):
    """ Build one fully-connected layer without activations. """
    with tf.variable_scope(name):
        W = tf.get_variable("W", shape=[in_size, out_size], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", shape=[1, out_size], dtype=tf.float32,
                            initializer=tf.constant_initializer())
    return tf.matmul(in_data, W) + b


def build_highway_layer(name, in_data, in_size):
    """ Build one fully-connected layer with skip-connection. """
    with tf.variable_scope(name):
        activation_in = build_linear_layer("activation", in_data, in_size,
                                           in_size)
        gate_in = build_linear_layer("gate", in_data, in_size, in_size)
    activation = tf.nn.relu(activation_in)
    gate = tf.nn.sigmoid(gate_in)
    return tf.multiply(gate, activation) + tf.multiply(1.0 - gate, in_data)


def build_highway_net(in_data, in_size, layers_count):
    """ Create highway network on top of the input. """
    with tf.variable_scope("highway"):
        for i in range(layers_count):
            output = build_highway_layer("layer_{}".format(i), in_data, in_size)
            in_data = output
    return output


def build_lstm_cell(layers_sizes, in_lstm_dropout):
    """ Build multi-layer lstm cell. """
    cells = []
    for i, layer_size in enumerate(layers_sizes):
        cell = tf.contrib.rnn.BasicLSTMCell(layer_size)
        if i != len(layers_sizes) - 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=1.0 - in_lstm_dropout
            )
        cells.append(cell)
    return tf.contrib.rnn.MultiRNNCell(cells)
