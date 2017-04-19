import tensorflow as tf
import h5py
import numpy as np
from tensorflow.contrib import rnn
#todo : hdf5 update file ?
# convolutional process 
# basic function : 2 convolutional layer 1 fully connected layer 
# data shape : 4*101*101
# the shape of raw data is still unk
# conv2d processs data with shape : [batch, height, width, channel]

def conv2d(x, w, b, strides):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)    

def maxpool2d(x, k=5):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# get 1000 final node for convultional layer
def conv_process(x, weights_conv, bias, dropout):
    x = conv2d(x, weights_conv['w1'], biases_conv['b1'], 1)
    x = maxpool2d(x)
    x = conv2d(x, weights_conv['w2'], biases_conv['b2'], 1)
    x = maxpool2d(x)
    x = tf.reshape(x, [-1, weights_conv['wf'].get_shape().as_list()[0]])  #  get 1000 final nodes 
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

weights_conv = {
    'w1':tf.Variable(tf.random_normal([5, 5, 4, 1024])),  # after maxpool2d 
    'w2':tf.Variable(tf.random_normal([5, 5, 1024, 512])),
    'wf':tf.Variable(tf.random_normal([5*5*512, 1000]))
}

biases_conv = {
    'b1':tf.Variable(tf.random_normal([1024])),
    'b2':tf.Variable(tf.random_normal([512])),
    'bf':tf.Variable(tf.random_normal([1000]))
}

weights_lstm = {
    'out': tf.Variable(tf.random_normal([128, 1]))
}

biases_lstm = {    
    'out': tf.Variable(tf.random_normal([1]))
}

# load weights_conv from the data stored in files format : pickle or hdf5
def load_params():
    weights_conv = {}
    biases_conv = {}
    h5f = h5py.File('./hdf5_data/hdf5_conv.h5', 'r')
    weights_conv['w1'] = h5f['w1']
    weights_conv['w2'] = h5f['w2']
    weights_conv['wf'] = h5f['wf']
    biases_conv['b1'] = h5f['b1']
    biases_conv['b2'] = h5f['b2']
    biases_conv['bf'] = h5f['bf']
    return weights_conv, biases_conv    

# stored the weights_conv and biases_conv into the hdf5 file 
def store_param():
    h5f = h5py.File('./hdf5_data/hdf5_conv.h5', 'w')
    h5f.create_dataset('w1', data=weights_conv['w1'])
    h5f.create_dataset('w2', data=weights_conv['w2'])    
    h5f.create_dataset('wf', data=weights_conv['wf'])
    h5f.create_dataset('b1', data=biases_conv['b1'])
    h5f.create_dataset('b2', data=biases_conv['b2'])
    h5f.create_dataset('bf', data=biases_conv['bf'])
    h5f.close()    

# update param for conv param 
def update_param():
    import os
    filename = './hdf5_data/hdf5_conv.h5'
    if os.path.exists(filename):
        os.remove(filename)
    store_param()
    pass