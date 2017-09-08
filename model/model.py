import tensorflow as tf
from data_utilities.data_processor import *


# TODO investigate what exactly these VARIABLE methods do
def get_conv_2d_layer(my_input, output_shape, k_height, k_width, name='conv'):
    # the method returns conv2d layer with specified parameters
    # w stands for the kernel and b for the bias, k for kernel
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_height, k_width,
                                  my_input.get_shape()[-1], output_shape])
        b = tf.get_variable('b', [output_shape])
        my_strides = [1, 1, 1, 1]

    return tf.nn.conv2d(my_input, w, strides=my_strides, padding='VALID') + b


def reshape_to_2d(my_input, max_word_len):
    # method reshapes multidimensional input into 2d
    # so that a conv2d layer appliance is valid
    output = tf.reshape(my_input, [-1, max_word_len, ALPHABET_LEN])
    output = tf.expand_dims(output, 1)
    return output


def softmax(my_input, output_shape, scope='softmax'):

    with tf.variable_scope(scope):
        w = tf.get_variable('w', [my_input.get_shape()[1], output_shape])
        b = tf.get_variable('b', [output_shape])

    return tf.nn.softmax(tf.matmul(my_input, w) + b)


# TODO more helper methods here



class Conv2LSTM():

    def __init__(self):
        # get params
        self.params = self.get_params()
        # placeholdere for sentences encode into 4dim tensors
        self.sentence = tf.placeholder('float32',
                                       shape=[None,
                                              None,
                                              self.params['MAX_WORD_LEN'],
                                              ALPHABET_LEN])
        # placeholder for sentiment label
        self.sent_label = tf.placeholder('float32', shape=[None, 2], name='sent')

    def get_params(self):
        return {
            'BATCH_SIZE': 64,
            'EPOCHS': 50,
            'MAX_WORD_LEN': 16,
            'LR': 0.0001,
            'PATIENCE': 1000,

        }


    def conv_char_embedding(self, my_input, kernels, kernel_num, scope='CharConv'):

        assert len(kernels) == len(kernel_num)

        resized_input = reshape_to_2d(my_input, self.max_word_len)

        conv_net = []
        with tf.variable_scope(scope):
            for kernel_shape, kernel_num_shape in zip(kernels, kernel_num):
                # TODO wtf?
                reduced_len = self.max_word_len - kernel_shape + 1

                # currently our input tensor is of size
                # [batch_size * sentence_length x max_word_length x alphabet_size x
                # kernel_feature_size]
                conv_layer = get_conv_2d_layer(my_input, kernel_num_shape, 1,
                                               kernel_shape,
                                               name='kernel %s' % kernel_shape)
                my_strides = [1, 1, 1, 1]
                max_pool = tf.nn.max_pool(tf.tanh(conv_layer),
                                             [1, 1, reduced_len, 1],
                                             strides=my_strides,
                                             padding='VALID')
                # TODO co to jest
                conv_net.append(tf.squeeze(max_pool, [1, 2]))

                # TODO co to za if i co robi concat
                if len(kernels) > 1:
                    output = tf.concat(conv_net, 1)
                else:
                    output = conv_net[0]
        return output

    def build_LSTM(self,
              training=False,
              testing_batch_size=1000,
              kernels=(1, 2, 3, 4, 5, 6, 7),
              kernel_quantity=(25, 50, 75, 100, 125, 150, 175),
              rnn_size=650,
              dropout=0.0,
              train_samples=1600000 * (1-VALIDATION_PERCENTAGE),
              valid_samples=1600000 * VALIDATION_PERCENTAGE):
        self.params = self.get_params()
        self.max_word_len = self.params['MAX_WORD_LEN']
        self.train_samples = train_samples
        self.valid_samples = valid_samples

        # if we are training use the training params else use the testing ones
        if training:
            self.BATCH_SIZE = self.params['BATCH_SIZE']
        else:
            self.BATCH_SIZE = testing_batch_size

        # here the magic starts to happen
        cnn = self.conv_char_embedding(self.sentence, kernels, kernel_quantity)
        # TODO: CZY NA PENO TAK MOZNA??? CHECKPOINT!!
        cnn = tf.reshape(cnn, [self.BATCH_SIZE, -1,
                               reduce(lambda x, y: x*y, kernel_quantity)])






