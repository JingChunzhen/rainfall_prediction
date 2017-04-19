import tensorflow as tf
import numpy as np
import h5py
import basic
import data_parse
from tensorflow.contrib import rnn

#weights_conv, biases_conv = basic.load_params()

# load and update weights and biases
# batch train use SGD 

def process(learning_rate):
    
    raw_x, raw_y = data_parse.preprocess(10)    
    weights_conv = basic.weights_conv
    biases_conv = basic.biases_conv

    weights_lstm = basic.weights_lstm
    biases_lstm = basic.biases_lstm

    x = tf.placeholder('float', [10, 101, 101, 4])
    y = tf.placeholder('float', [10, 1])
    conv_x = basic.conv_process(x, weights_conv, biases_conv, dropout=0.75)

    x_lstm = tf.placeholder('float', [15, 10, 1000])    
    pred = basic.lstm_process(x_lstm, weights_lstm, biases_lstm)

    RMSE = tf.sqrt(tf.reduce_mean(tf.square(raw_y - pred))) # wrt raw_y

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(RMSE)

    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:  # a loop supposed to be here 
        sess.run(init)
        
        train_x = []
        for i in range(15):
            _x = sess.run([conv_x], feed_dict={x:raw_x[i]})
            train_x.append(_x)
        train_x = np.asarray(train_x)        
        print(train_x.shape)  # get 15 1 10 1000
        train_x = np.transpose(train_x, [1, 0, 2, 3])
        print(train_x.shape)
        print(train_x[0].shape)  # get 15 10 1000

        train_x = np.asarray(train_x[0])
        raw_y = np.reshape(raw_y, [10, 1])
        feed_dict = {x_lstm:train_x, y:raw_y}  
        RMSE, _ = sess.run([RMSE, optimizer], feed_dict=feed_dict)          
        print(RMSE)
        pass
    pass

if __name__ == '__main__':    
    process(0.01)
    pass



