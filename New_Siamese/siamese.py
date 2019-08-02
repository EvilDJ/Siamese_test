import tensorflow as tf
import numpy as np

net_data = np.load('./my_net.npy',encoding = "latin1").item()
tf.reset_default_graph()

def siamese(xl, xr,sess):
    with tf.variable_scope('conv1'):
        w0 = net_data['Conv1/weights']
        b0 = net_data['Conv1/biases']
        # w_0 = tf.get_variable('w', initializer=tf.constant(w0))
        # b_0 = tf.get_variable('b', initializer=tf.constant(b0))
        convl = tf.nn.conv2d(xl, w0, [1, 2, 2, 1], 'VALID')
        convl = tf.add(convl, b0)
        convl_0 = tf.nn.relu(convl)
        convr = tf.nn.conv2d(xr, w0, [1, 2, 2, 1], 'VALID')
        convr = tf.add(convr, b0)
        convr_0 = tf.nn.relu(convr)
        pass
    with tf.variable_scope('conv2'):
        w1 = net_data['Conv2/weights']
        b1 = net_data['Conv2/biases']
        # w_1 = tf.get_variable('w', initializer=tf.constant(w1))
        # b_1 = tf.get_variable('b', initializer=tf.constant(b1))
        convl = tf.nn.conv2d(convl_0, w1, [1, 1, 1, 1], 'VALID')
        convl = tf.add(convl, b1)
        convl_1 = tf.nn.relu(convl)
        convr = tf.nn.conv2d(convr_0, w1, [1, 1, 1, 1], 'VALID')
        convr = tf.add(convr, b1)
        convr_1 = tf.nn.relu(convr)
        pass
    with tf.variable_scope('conv3'):
        w2 = net_data['Conv3/weights']
        b2 = net_data['Conv3/biases']
        # w = tf.get_variable('w', initializer=tf.constant(w2))
        # b = tf.get_variable('b', initializer=tf.constant(b2))
        convl = tf.nn.conv2d(convl_1, w2, [1, 1, 1, 1], 'SAME')
        convl = tf.add(convl, b2)
        convl_2 = tf.nn.relu(convl)
        convr = tf.nn.conv2d(convr_1, w2, [1, 1, 1, 1], 'SAME')
        convr = tf.add(convr, b2)
        convr_2 = tf.nn.relu(convr)
        pass
    with tf.variable_scope('maxpool1'):
        convl_2 = tf.nn.max_pool(convl_2, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
        convr_2 = tf.nn.max_pool(convr_2, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')

    with tf.variable_scope('conv4'):
        w3 = net_data['Conv4/weights']
        b3 = net_data['Conv4/biases']
        # w = tf.get_variable('w', initializer=tf.constant(w3))
        # b = tf.get_variable('b', initializer=tf.constant(b3))
        convl = tf.nn.conv2d(convl_2, w3, [1, 2, 2, 1], 'VALID')
        convl = tf.add(convl, b3)
        convl_3 = tf.nn.relu(convl)
        convr = tf.nn.conv2d(convr_2, w3, [1, 2, 2, 1], 'VALID')
        convr = tf.add(convr, b3)
        convr_3 = tf.nn.relu(convr)
        pass
    with tf.variable_scope('conv5'):
        w4 = net_data['Conv5/weights']
        b4 = net_data['Conv5/biases']
        # w = tf.get_variable('w', initializer=tf.constant(w4))
        # b = tf.get_variable('b', initializer=tf.constant(b4))
        convl = tf.nn.conv2d(convl_3, w4, [1, 1, 1, 1], 'SAME')
        convl = tf.add(convl, b4)
        convl_4 = tf.nn.relu(convl)
        convr = tf.nn.conv2d(convr_3, w4, [1, 1, 1, 1], 'SAME')
        convr = tf.add(convr, b4)
        convr_4 = tf.nn.relu(convr)
        pass
    with tf.variable_scope('conv6'):
        w5 = net_data['Conv6/weights']
        b5 = net_data['Conv6/biases']
        # w = tf.get_variable('w', initializer=tf.constant(w5))
        # b = tf.get_variable('b', initializer=tf.constant(b5))
        convl = tf.nn.conv2d(convl_4, w5, [1, 1, 1, 1], 'VALID')
        convl = tf.add(convl, b5)
        convl_5 = tf.nn.relu(convl)
        convr = tf.nn.conv2d(convr_4, w5, [1, 1, 1, 1], 'VALID')
        convr = tf.add(convr, b5)
        convr_5 = tf.nn.relu(convr)
        pass
    with tf.variable_scope('maxpool2'):
        convl_5 = tf.nn.max_pool(convl_5, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
        convr_5 = tf.nn.max_pool(convr_5, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
    distance = np.linalg.norm(sess.run(convl_5)-sess.run(convr_5))
    return distance

def match(distance ,dela):
    if distance <= dela :
        boo =0
    else: boo = 1
    return boo
