import tensorflow as tf
from tensorflow.contrib import rnn


# conv process
# basic function : 2 convolutional layer + 1 fully connected layer
# data shape : 4*101*101
# conv2d processs data with shape : [batch, height, width, channel]
# return data shape : 1000 dtype : float

# lstm process
# basic functon : 1 lstm layer + 1 linear regression layer
# data shape : 15*1000
# return data shape : 1 dtype : float

def conv2d(x, w, b, strides):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=5):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# get 1000 final node for convolutional layer


def conv_process(x, weights_conv, biases_conv, dropout):
    x = conv2d(x, weights_conv['w1'], biases_conv['b1'], 1)
    x = maxpool2d(x)
    x = conv2d(x, weights_conv['w2'], biases_conv['b2'], 1)
    x = maxpool2d(x)
    # get 1000 final nodes
    x = tf.reshape(x, [-1, weights_conv['wf'].get_shape().as_list()[0]])
    x = tf.add(tf.matmul(x, weights_conv['wf']), biases_conv['bf'])
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, dropout)
    return x


def lstm_process(x, weights_lstm, biases_lstm):
    x = tf.reshape(x, [-1, 1000])
    x = tf.split(value=x, num_or_size_splits=15, axis=0)
    lstm_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, inputs=x, dtype='float')
    return tf.add(tf.matmul(outputs[-1], weights_lstm['out']), biases_lstm['out'])
