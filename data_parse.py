import tensorflow as tf
import tflearn
import h5py
import numpy as np

# split data use command in linux : split
# train model use batch data
# get the splited raw data from file 
# CIKM data shape : 
'''
batch ->
    id0, label0, radar_map0
    id1, label1, radar_map1
    ...
    idn, labeln, radar_mapn

radar_map ->
    shape: 15*4*101*101
'''
# linux command cat train.txt | head 100 > sample 
# each line get a data for id, label, and radar_map data type : str
# the command above get a sample which has 100 data of data file 

# parse the data get from the sample 

def parse_data(batch_size):
    filename = './data/CIKM2017_train/data_new/CIKM2017_train/sample'
    with open(filename, 'r') as f:
        raw_x = []
        raw_y = []
        count = 0
        #for line in f.readlines():            
        for line in f:
            id, label, radar_map = line.split(',')  # type(radar_map) == str                 
            radar_map = radar_map.split(' ') # get 612060
            raw_y.append(label)
            raw_x.append(radar_map)
            count += 1
            if count == batch_size:
                break
    return raw_x, raw_y

# raw_x need to be normalized see tensorflow wrt normalization 
def preprocess(batch_size):
    raw_x, raw_y = parse_data(batch_size)
    raw_x = np.asarray(raw_x, dtype=np.float)    
    raw_y = np.asarray(raw_y, dtype=np.float)    
    raw_x = np.reshape(raw_x, [batch_size, 15, 4, 101, 101])  # 0, 1, 2, 3, 4
    raw_x = np.transpose(raw_x, [1, 0, 3, 4, 2])    
    return raw_x, raw_y
    pass

if __name__ == '__main__':     
    '''           
    raw_x, raw_y = preprocess(10) # raw_x.shape: (15, 10, 101, 101, 4)
    print(raw_x[0].shape)  # shape: (10, 101, 101, 4)        
    print(raw_y.shape)     
    '''
    import time
    start = time.time()
    raw_x, raw_y = preprocess(10)
    end = time.time()
    print(raw_x.shape)
    print(end - start)
    

    
    

            
        

