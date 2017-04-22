import linecache
import os

import h5py
import numpy as np
import tensorflow as tf

import basic

# split data use command in linux : split
# train model use batch data
# get the splited raw data from file
# CIKM data shape :
# batch ->
#     id0, label0, radar_map0
#     id1, label1, radar_map1
#     ...
#     idn, labeln, radar_mapn

# radar_map ->
#     shape: 15*4*101*101
# linux command cat train.txt | head -100 > sample
# each line get a data for id, label, and radar_map data type : str
# the command above get a sample which has 100 data of data file
# convert the data type int <-> str and ASCII int <-> str


def get_dir(dir_count):
    filename = './data/CIKM2017_train/split_file'
    if dir_count < 10:
        filename += '0'
        filename += str(dir_count)
    else:
        filename += str(dir_count)
    return filename

# get raw_x and  raw_y


# dir_count count from 0, batch_count count from 0
def generate_batch(dir_count, batch_count, batch_size):
    filename = get_dir(dir_count)

    raw_x = []
    raw_y = []

    for i in range(1, batch_size + 1):
        line_num = batch_count * batch_size + i  # line num count from 1
        line = linecache.getline(filename, line_num)  # cant read num 0 line
        id, label, radar_map = line.split(',')
        radar_map = radar_map.split(' ')  # get 612060
        raw_y.append(label)
        raw_x.append(radar_map)

    raw_x = np.asarray(raw_x, dtype=np.float)
    raw_y = np.asarray(raw_y, dtype=np.float)

    raw_x = np.reshape(raw_x, [batch_size, 15, 4, 101, 101])
    raw_x = np.transpose(raw_x, [1, 0, 3, 4, 2])
    raw_y = np.reshape(raw_y, [batch_size, 1])

    return raw_x, raw_y


# load and store weights and biases data to the hdf5


def load_params():
    if is_empty():
        return basic.weights_conv, basic.biases_conv, basic.weights_lstm, basic.biases_lstm
    else:
        weights_conv = {}
        biases_conv = {}

        weights_lstm = {}
        biases_lstm = {}

        h5f = h5py.File('./hdf5_data/hdf5_data.h5', 'r')

        weights_conv['w1'] = h5f['w1']
        weights_conv['w2'] = h5f['w2']
        weights_conv['wf'] = h5f['wf']

        biases_conv['b1'] = h5f['b1']
        biases_conv['b2'] = h5f['b2']
        biases_conv['bf'] = h5f['bf']

        weights_lstm['out'] = h5f['weights_out']
        biases_lstm['out'] = h5f['biases_out']

        h5f.close()

        return weights_conv, biases_conv, weights_lstm, biases_lstm


def is_empty():
    filename = './hdf5_data/hdf5_data.h5'
    flag = True
    if os.path.exists(filename):
        if os.path.getsize(filename):  # != 0
            flag = False
        else:
            flag = True
    else:
        flag = True

    return flag


# stored conv params and lstm params into the hdf5 file


def store_param(weights_conv, biases_conv, weights_lstm, biases_lstm):
    h5f = h5py.File('./hdf5_data/hdf5_data.h5', 'w')

    h5f.create_dataset('w1', data=weights_conv['w1'])
    h5f.create_dataset('w2', data=weights_conv['w2'])
    h5f.create_dataset('wf', data=weights_conv['wf'])

    h5f.create_dataset('b1', data=biases_conv['b1'])
    h5f.create_dataset('b2', data=biases_conv['b2'])
    h5f.create_dataset('bf', data=biases_conv['bf'])

    h5f.create_dataset('weights_out', data=weights_lstm['out'])
    h5f.create_dataset('biases_out', data=biases_lstm['out'])

    h5f.close()

# update params


def update_param(weights_conv, biases_conv, weights_lstm, biases_lstm):
    filename = './hdf5_data/hdf5_data.h5'

    if os.path.exists(filename):
        os.remove(filename)
    store_param(weights_conv, biases_conv, weights_lstm, biases_lstm)
    pass
