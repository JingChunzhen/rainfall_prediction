import linecache
import os

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
# need normalization and rid of the dirty data
# check label == -1
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

    # normalization
    raw_x = raw_x / 203
    return raw_x, raw_y


def generate_test_batch(batch_count, batch_size, test_name):
    if test_name is 'validation':
        filename = './data/CIKM2017_testA/validation_set'
    else:
        dir_count = np.random.randint(0, 16)
        if dir_count < 10:
            filename = './data/CIKM2017_testA/split_data0' + str(dir_count)
        else:
            filename = './data/CIKM2017_testA/split_data' + str(dir_count)

    raw_x = []
    raw_y = []

    for i in range(1, batch_size + 1):
        line_num = batch_count * batch_size + i
        line = linecache.getline(filename, line_num)
        id, label, radar_map = line.split(',')
        radar_map = radar_map.split(' ')
        raw_y.append(label)
        raw_x.append(radar_map)

    raw_x = np.asarray(raw_x, dtype=np.float)
    raw_y = np.asarray(raw_y, dtype=np.float)

    raw_x = np.reshape(raw_x, [batch_size, 15, 4, 101, 101])
    raw_x = np.transpose(raw_x, [1, 0, 3, 4, 2])
    raw_y = np.reshape(raw_y, [batch_size, 1])

    raw_x = raw_x / 203
    return raw_x, raw_y
