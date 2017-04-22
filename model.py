import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import basic
import data_parse

# split the data into 32 parts each part has 100 lines of data
# get each batch in each split data
# update params once the split data training over
# then the validation test commence
# and the cycle is repeated till the epochs end

global dir_num = 32

weights_conv = {
    'w1': tf.Variable(tf.random_normal([5, 5, 4, 1024])),  # after maxpool2d
    'w2': tf.Variable(tf.random_normal([5, 5, 1024, 512])),
    'wf': tf.Variable(tf.random_normal([5 * 5 * 512, 1000]))
}

biases_conv = {
    'b1': tf.Variable(tf.random_normal([1024])),
    'b2': tf.Variable(tf.random_normal([512])),
    'bf': tf.Variable(tf.random_normal([1000]))
}

weights_lstm = {
    'out': tf.Variable(tf.random_normal([128, 1]))
}

biases_lstm = {
    'out': tf.Variable(tf.random_normal([1]))
}

model_path = './model_data/model.ckpt'

def process(learning_rate, batch_size, epochs):
    assert 100 % batch_size == 0
    assert batch_size <= 100    

    x = tf.placeholder('float', [batch_size, 101, 101, 4])
    y = tf.placeholder('float', [batch_size, 1])
    x_conv = basic.conv_process(x, weights_conv, biases_conv, dropout=0.75)

    x_lstm = tf.placeholder('float', [15, batch_size, 1000])
    pred = basic.lstm_process(x_lstm, weights_lstm, biases_lstm)

    RMSE = tf.sqrt(tf.reduce_mean(tf.square(raw_y - pred)))  # wrt raw_y

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(RMSE)
        
    init = tf.initialize_all_variables()
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        batch_num = 100 / batch_size
        epoch = 0

        if os.path.getsize('./model_data/model.ckpt') != 0:
            saver.restore(sess, './model_data/model.ckpt')
        
        while epoch < epochs:
            for dir_count in range(dir_num):
                for batch_count in range(batch_num):

                    raw_x, raw_y = data_parse.generate_batch(
                        dir_count, batch_count, batch_size)

                    train_x = []

                    for i in range(15):
                        _x = sess.run([x_conv], feed_dict={x: raw_x[i]})
                        train_x.append(_x)

                    train_x = np.asarray(train_x)
                    print(train_x.shape)  # get 15 1 10 1000
                    train_x = np.transpose(train_x, [1, 0, 2, 3])

                    print(train_x.shape)
                    print(train_x[0].shape)  # get 15 10 1000

                    train_x = np.asarray(train_x[0])

                    RMSE, _ = sess.run(
                        [RMSE, optimizer],
                        feed_dict={x_lstm: train_x, y: raw_y}
                    )
                    print(RMSE)

                # validation
                # data_parse.update_param(
                #     weights_conv, biases_conv, weights_lstm, biases_lstm)
                model_path = './model_data/'
                saver.save(sess, model_path)
                # validation test
                pass  # end of for
            epoch += 1
        pass  # end of while


if __name__ == '__main__':
    process(learning_rate=0.01, batch_size=10, epochs=1)
    pass
