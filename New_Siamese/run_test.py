import tensorflow as tf
import numpy as np
import random
slim = tf.contrib.slim


my_net_path = './my_net.npy'
net_data = np.load(my_net_path,'rb',encoding='latin1').item()
tf.reset_default_graph()
xl = tf.placeholder(tf.float32,shape=[1,225,225,3],name='pic_l')
xr = tf.placeholder(tf.float32,shape=[1,225,225,3],name='pic_r')

def produce_threshold():
    threshold = random.sample(range(0, 100), 100)
    thre_ = []
    for i in threshold:
        thre_.append(float(i/100))
    return thre_

def dist(out1,out2):
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1 - out2) + 1e-10))
    e_w = tf.reduce_mean(E_w)
    return e_w

def siamese(xl,xr):
    with tf.variable_scope('conv1'):
        w0, b0 = net_data['Conv1']
        w = tf.get_variable('w', initializer=tf.constant(w0))
        b = tf.get_variable('b', initializer=tf.constant(b0))
        convl = tf.nn.conv2d(xl, w, [1, 2, 2, 1], 'VALID')
        convl = tf.add(convl, b)
        convl_0 = tf.nn.relu(convl)
        convr = tf.nn.conv2d(xr, w, [1, 2, 2, 1], 'VALID')
        convr = tf.add(convr, b)
        convr_0 = tf.nn.relu(convr)
        pass
    with tf.variable_scope('conv2'):
        w1, b1 = net_data['Conv2']
        w = tf.get_variable('w', initializer=tf.constant(w1))
        b = tf.get_variable('b', initializer=tf.constant(b1))
        convl = tf.nn.conv2d(convl_0, w, [1, 1, 1, 1], 'VALID')
        convl = tf.add(convl, b)
        convl_1 = tf.nn.relu(convl)
        convr = tf.nn.conv2d(convr_0, w, [1, 1, 1, 1], 'VALID')
        convr = tf.add(convr, b)
        convr_1 = tf.nn.relu(convr)
        pass
    with tf.variable_scope('conv3'):
        w2, b2 = net_data['Conv3']
        w = tf.get_variable('w', initializer=tf.constant(w2))
        b = tf.get_variable('b', initializer=tf.constant(b2))
        convl = tf.nn.conv2d(convl_1, w, [1, 1, 1, 1], 'SAME')
        convl = tf.add(convl, b)
        convl_2 = tf.nn.relu(convl)
        convr = tf.nn.conv2d(convr_1, w, [1, 1, 1, 1], 'SAME')
        convr = tf.add(convr, b)
        convr_2 = tf.nn.relu(convr)
        pass
    with tf.variable_scope('maxpool1'):
        convl_2 = tf.nn.max_pool(convl_2, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
        convr_2 = tf.nn.max_pool(convr_2, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')

    with tf.variable_scope('conv4'):
        w3, b3 = net_data['Conv4']
        w = tf.get_variable('w', initializer=tf.constant(w3))
        b = tf.get_variable('b', initializer=tf.constant(b3))
        convl = tf.nn.conv2d(convl_2, w, [1, 2, 2, 1], 'VALID')
        convl = tf.add(convl, b)
        convl_3 = tf.nn.relu(convl)
        convr = tf.nn.conv2d(convr_2, w, [1, 2, 2, 1], 'VALID')
        convr = tf.add(convr, b)
        convr_3 = tf.nn.relu(convr)
        pass
    with tf.variable_scope('conv5'):
        w4, b4 = net_data['Conv5']
        w = tf.get_variable('w', initializer=tf.constant(w4))
        b = tf.get_variable('b', initializer=tf.constant(b4))
        convl = tf.nn.conv2d(convl_3, w, [1, 1, 1, 1], 'SAME')
        convl = tf.add(convl, b)
        convl_4 = tf.nn.relu(convl)
        convr = tf.nn.conv2d(convr_3, w, [1, 1, 1, 1], 'SAME')
        convr = tf.add(convr, b)
        convr_4 = tf.nn.relu(convr)
        pass
    with tf.variable_scope('conv6'):
        w5, b5 = net_data['Conv2']
        w = tf.get_variable('w', initializer=tf.constant(w5))
        b = tf.get_variable('b', initializer=tf.constant(b5))
        convl = tf.nn.conv2d(convl_4, w, [1, 1, 1, 1], 'VALID')
        convl = tf.add(convl, b)
        convl_5 = tf.nn.relu(convl)
        convr = tf.nn.conv2d(convr_4, w, [1, 1, 1, 1], 'VALID')
        convr = tf.add(convr, b)
        convr_5 = tf.nn.relu(convr)
        pass
    with tf.variable_scope('maxpool2'):
        convl_5 = tf.nn.max_pool(convl_5, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
        convr_5 = tf.nn.max_pool(convr_5, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')

    return convl_5,convr_5