import tensorflow as tf
import numpy as np
import pandas as pd


def spp_layer(input_, levels=4, name='SPP_layer', pool_type='max_pool'):
    '''
    Multiple Level SPP layer.

    Works for levels=[1, 2, 3, 6].
    '''
    
    shape = input_.get_shape().as_list()
    
    with tf.variable_scope(name):
        
        for l in range(levels):
            
            l = l + 1
            ksize = [1, np.ceil(shape[1] / l + 1).astype(np.int32), np.ceil(shape[2] / l + 1).astype(np.int32), 1]
            
            strides = [1, np.floor(shape[1] / l + 1).astype(np.int32), np.floor(shape[2] / l + 1).astype(np.int32), 1]
            
            if pool_type == 'max_pool':
                pool = tf.nn.max_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool, (shape[0], -1), )
            
            else:
                pool = tf.nn.avg_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool, (shape[0], -1))
            print("Pool Level {:}: shape {:}".format(l, pool.get_shape().as_list()))
            if l == 1:
                
                x_flatten = tf.reshape(pool, (shape[0], -1))
            else:
                x_flatten = tf.concat((x_flatten, pool), axis=1)
            print("Pool Level {:}: shape {:}".format(l, x_flatten.get_shape().as_list()))
            # pool_outputs.append(tf.reshape(pool, [tf.shape(pool)[1], -1]))
    
    return x_flatten