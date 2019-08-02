import tensorflow as tf
import numpy as np

def spp(data,bins):
    shape = data.get_shape().as_list()
    batches = shape[0]
    pool_out = []
    for i in range(bins):
        i = i+1
        kize = [1, np.ceil(shape[1]/ i + 1).astype(np.int32), np.ceil(shape[1]/ i + 1).astype(np.int32), 1]
        stride = [1, np.floor(shape[1]/i + 1).astype(np.int32), np.floor(shape[2]/i +1).astype(np.int32), 1]
        pool_out.append(tf.nn.max_pool(data, ksize=kize, strides=stride, padding='SAME'))
        print("Pool level {:}: {:} ".format(i, kize))
    for i in range(bins):
        print("Pool Level {:}: shape {:}".format(i, pool_out[i].get_shape().as_list()))
        pool_out[i] = tf.reshape(pool_out[i],[batches,-1])
        print("Pool Level {:}: shape {:}".format(i, pool_out[i].get_shape().as_list()))
    output = tf.concat(axis=1, values=[pool_out[0], pool_out[1], pool_out[2]], name='spp_layer')
    return output
    #print(pool_out[i])
    #output = tf.concat(axis=1, values=[pool_out[0],pool_out[1],pool_out[2]],name='spp_layer')
    