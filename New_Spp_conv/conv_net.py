from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import spp_layer

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
tf.reset_default_graph()


def print_actication(t):
    print((t.op.name), ' ', t.get_shape().as_list())


"""
# siamese网络的loss值
def siamese_loss(out1, out2, y, Q=5):
    Q = tf.constant(Q, name="Q", dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1 - out2), 1))#sqrt求平方根，square求平方，reduce_sum(x,0,1)
    pos = tf.multiply(tf.multiply(y, 2 / Q), tf.square(E_w))#multiply相乘
    neg = tf.multiply(tf.multiply(1 - y, 2 * Q), tf.exp(-2.77 / Q * E_w))
    loss = pos + neg
    loss = tf.reduce_mean(loss)#reduce_mean按照某个维度求均值。
    return loss
    :param inputs:
    :param drop_keep_prob:
    :param final_endpoint:
    :param min_depth:
    :param depth_multiplier:
    :param scope:
    :return:
网络模型结构：
     Old name          | New name
  =======================================
  conv0             | Conv2d_1a_3x3
  batch_normal+pool
  conv1             | Conv2d_2a_3x3
  batch_normal+pool
  conv2             | Conv2d_2b_3x3
  batch_normal+pool
  pool1             | MaxPool_3a_3x3
  conv3             | Conv2d_3b_1x1
  batch_normal+pool
  conv4             | Conv2d_4a_3x3
  batch_normal+pool
  pool2             | MaxPool_5a_3x3
  mixed_35x35x256a  | Mixed_5b
  mixed_35x35x288a  | Mixed_5c
  mixed_35x35x288b  | Mixed_5d
  batch_normal+pool
  mixed_17x17x768a  | Mixed_6a
  mixed_17x17x768b  | Mixed_6b
  mixed_17x17x768c  | Mixed_6c
  mixed_17x17x768d  | Mixed_6d
  mixed_17x17x768e  | Mixed_6e
  batch_normal+pool
  mixed_8x8x1280a   | Mixed_7a
  mixed_8x8x2048a   | Mixed_7b
  mixed_8x8x2048b   | Mixed_7c
  spp_layer
  fc
    """
# inputs, keep_prob, drop_keep_prob, loss, lr, iterarions, batch_size,  设置图片格式为225*225
def siamese(inputs, drop_keep_prob=0.8, min_depth=16, depth_multiplier=1.0, scope=None):
    # Inception_V3 网络内体架构
    print_actication(inputs)
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
            # 225 * 225 * 3
            net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope='Conv2d_1a_3x3')
            # print_actication(net)
            # 112 * 112 * 32
            net = slim.conv2d(net, depth(32), [3, 3], scope='Conv2d_2a_3x3')
            # print_actication(net)
            # 110 * 110 * 32
            net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
            # print_actication(net)
            # 110 * 110 * 64
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
            # print_actication(net)
            # 56 * 56 * 64
            net = slim.conv2d(net, depth(80), [1, 1], scope='Conv2d_3b_1x1')
            # print_actication(net)
            # 56 * 56 * 80
            net = slim.conv2d(net, depth(192), [3, 3], scope='Conv2d_4a_3x3')
            # print_actication(net)
            # 54 * 54 * 192
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')
            # print_actication(net)
            # 27 * 27 * 192
        # Inception blocks Inception核心模块
        # with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):#设置补零
        #     with tf.variable_scope('Mixed_5b'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_1 = slim.conv2d(branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        #             branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
        #         with tf.variable_scope('Branch_3'):
        #             branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        #             branch_3 = slim.conv2d(branch_3, depth(32), [1, 1], scope='Conv2d_0b_1x1')
        #         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        #      #   print_actication(net)
        #     # mixed_0: 35 x 35 x 256.
        #     with tf.variable_scope('Mixed_5c'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
        #             branch_1 = slim.conv2d(branch_1, depth(64), [5, 5], scope='Conv_1_0c_5x5')
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        #             branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
        #         with tf.variable_scope('Branch_3'):
        #             branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        #             branch_3 = slim.conv2d(branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
        #         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        #       #  print_actication(net)
        #     # mixed_1: 35 x 35 x 288.
        #     with tf.variable_scope('Mixed_5d'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_1 = slim.conv2d(branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        #             branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
        #         with tf.variable_scope('Branch_3'):
        #             branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        #             branch_3 = slim.conv2d(branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
        #         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        #        # print_actication(net)
        #     # mixed_2: 35 x 35 x 288.
        #     with tf.variable_scope('Mixed_6a'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        #             branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
        #         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
        #         #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_6a')
        #         #print_actication(net)
        #     # mixed_3: 13 * 13 * 672.
        #     with tf.variable_scope('Mixed_6b'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_1 = slim.conv2d(branch_1, depth(128), [1, 7], scope='Conv2d_0b_1x7')
        #             branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_2 = slim.conv2d(branch_2, depth(128), [7, 1], scope='Conv2d_0b_7x1')
        #             branch_2 = slim.conv2d(branch_2, depth(128), [1, 7], scope='Conv2d_0c_1x7')
        #             branch_2 = slim.conv2d(branch_2, depth(128), [7, 1], scope='Conv2d_0d_7x1')
        #             branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
        #         with tf.variable_scope('Branch_3'):
        #             branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        #             branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        #         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        #         #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_6b')
        #         #print_actication(net)
        #     # mixed_4: 13 x 13 x 768.
        #     with tf.variable_scope('Mixed_6c'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_1 = slim.conv2d(branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
        #             branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
        #             branch_2 = slim.conv2d(branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
        #             branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
        #             branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
        #         with tf.variable_scope('Branch_3'):
        #             branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        #             branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        #         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        #         #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_6c')
        #         #print_actication(net)
        #     # mixed_5: 13 x 13 x 768.
        #     with tf.variable_scope('Mixed_6d'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_1 = slim.conv2d(branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
        #             branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
        #             branch_2 = slim.conv2d(branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
        #             branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
        #             branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
        #         with tf.variable_scope('Branch_3'):
        #             branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        #             branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        #         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        #         #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_6d')
        #         #print_actication(net)
        #     # mixed_6: 13 * 13 * 768.
        #     with tf.variable_scope('Mixed_6e'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_1 = slim.conv2d(branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
        #             branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_2 = slim.conv2d(branch_2, depth(192), [7, 1], scope='Conv2d_0b_7x1')
        #             branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0c_1x7')
        #             branch_2 = slim.conv2d(branch_2, depth(192), [7, 1], scope='Conv2d_0d_7x1')
        #             branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
        #         with tf.variable_scope('Branch_3'):
        #             branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        #             branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        #         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        #         #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_6e')
        #         #print_actication(net)
        #     # mixed_7: 13 x 13 x 768.
        #     with tf.variable_scope('Mixed_7a'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_1 = slim.conv2d(branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
        #             branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        #             branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
        #         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
        #         #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_7a')
        #         #print_actication(net)
        #     # mixed_8: 6 * 6 * 1280.
        #     with tf.variable_scope('Mixed_7b'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_1 = tf.concat(axis=3, values=[
        #                 slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
        #                 slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_2 = slim.conv2d(
        #                 branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
        #             branch_2 = tf.concat(axis=3, values=[
        #                 slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
        #                 slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
        #         with tf.variable_scope('Branch_3'):
        #             branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        #             branch_3 = slim.conv2d(
        #                 branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        #         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        #         #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_7b')
        #         #print_actication(net)
        #     # mixed_9: 6 x 6 x 2048.
        #     with tf.variable_scope('Mixed_7c'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_1 = tf.concat(axis=3, values=[
        #                 slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
        #                 slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_2 = slim.conv2d(
        #                 branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
        #             branch_2 = tf.concat(axis=3, values=[
        #                 slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
        #                 slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
        #         with tf.variable_scope('Branch_3'):
        #             branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
        #             branch_3 = slim.conv2d(
        #                 branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        #         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        #         #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_7c')
        #         #print_actication(net)
    # Spp_layer金字塔池化层，保证任意尺度的图片输入,box spplayer内核个数
    box = 3
    net = spp_layer.spp(net, box)
    # print_actication(net)
    # 建立3个全连接层 AvgPool层，两个Conv层，一个Full_connect层，一个Softmax层,Final pooling and prediction
    # 第一个全连接层
    with tf.name_scope('fc1') as scope:
        w_fc1 = tf.Variable(tf.truncated_normal(shape=[int(net.get_shape()[1]), 2014], stddev=0.05, mean=0),
                            name='w_fc1')
        b_fc1 = tf.Variable(tf.zeros(2014), name='b_fc1')
        fc1 = tf.add(tf.matmul(net, w_fc1), b_fc1)
        # print_actication(fc1)
    with tf.name_scope('relu_fc1') as scope:
        relu_fc1 = tf.nn.relu(fc1, name='relu_fc1')
    with tf.name_scope('drop_1') as scope:
        drop_1 = tf.nn.dropout(relu_fc1, keep_prob=drop_keep_prob, name='drop_1')
        # print_actication(drop_1)
    # with tf.name_scope('bn_fc1') as scope:
    #   bn_fn1 = tf.layers.batch_normalization(drop_1, name='bn_fc1')
    # print_actication(drop_1)
    # 第二个全连接层
    with tf.name_scope('fc2') as scope:
        w_fc2 = tf.Variable(tf.truncated_normal(shape=[2014, 512], stddev=0.05, mean=0), name='w_fc2')
        b_fc2 = tf.Variable(tf.zeros(512), name='b_fc2')
        fc2 = tf.add(tf.matmul(drop_1, w_fc2), b_fc2)
    with tf.name_scope('relu_fc2') as scope:
        relu_fc2 = tf.nn.relu(fc2, name='relu_fc2')
    with tf.name_scope('drop_2') as scope:
        drop_2 = tf.nn.dropout(relu_fc2, keep_prob=drop_keep_prob, name='drop_2')
    # with tf.name_scope('bn_fc2') as scope:
    #   bn_fn2 = tf.layers.batch_normalization(drop_2, name='bn_fc2')
    # print_actication(drop_2)
    # 第三层全连接层
    with tf.name_scope('fc3') as scope:
        w_fc3 = tf.Variable(tf.truncated_normal(shape=[512, 50], stddev=0.05, mean=0), name='w_fc3')
        b_fc3 = tf.Variable(tf.zeros(50), name='b_fc3')
        fc3 = tf.add(tf.matmul(drop_2, w_fc3), b_fc3)
    with tf.name_scope('relu_fc3') as scope:
        relu_fc3 = tf.nn.relu(fc3, name='relu_fc2')
    with tf.name_scope('drop_3') as scope:
        drop_3 = tf.nn.dropout(relu_fc3, keep_prob=drop_keep_prob, name='drop_2')
    # print_actication(drop_3)
    # with tf.name_scope('fc4') as scope:
    #     w_fc4 = tf.Variable(tf.truncated_normal(shape=[50, 2], stddev=0.05, mean=0), name='w_fc4')
    #     b_fc4 = tf.Variable(tf.zeros(2), name='b_fc4')
    #     fc4 = tf.add(tf.matmul(drop_3,w_fc4), b_fc4)
    return drop_3
    # drop_3 得出高维特征，fc4 是否匹配


def dist(out1, out2):
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1 - out2 + 1e-10)))  # sqrt求平方根，square求平方，reduce_sum(x,0,1)
    distance = tf.reduce_mean(E_w)
    return distance


# siamese网络的loss值和两张图片高维特征欧氏距离
def loss(out1, out2, y, Q=5.0):
    # out1, out2, 两张图片的高维特征，y为out1 & out2的类别
    # 输出：网络的loss值以及欧氏距离
    Q = tf.constant(Q, name="Q", dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1 - out2 + 1e-10)))  # sqrt求平方根，square求平方，reduce_sum(x,0,1)
    pos = tf.multiply(tf.multiply((1 - y), 2 / Q), tf.square(E_w))  # multiply相乘
    neg = tf.multiply(tf.multiply(y, 2 * Q), tf.exp(-2.77 * E_w / Q))
    loss = pos + neg
    loss = tf.reduce_mean(loss)  # reduce_mean按照某个维度求均值。
    distance = tf.reduce_mean(E_w)
    return loss, distance


# 利用阈值点判断判断两张图片是否匹配
def match(distance, dela):
    # dela阈值，距离小于阈值判断为匹配，距离大于阈值判断为不匹配，0代表匹配，1代表不匹配
    if distance <= dela:
        boo = 0
    else:
        boo = 1
    return boo


# 网络评价准确率,以ROC为标准
def evaluation(fp, tp, tn, fn):
    # X=FP,Y=TN,M=TP,N=FN
    with tf.variable_scope('accuracy') as scope:
        # FPR = float(fp/(fp+tn))
        # TPR = float(tp/(tp+fn))
        # accuracy = tf.cast(float((fn+tp)/(fp+tp+fn+tn)),tf.float32)
        # tf.summary.scalar(scope.name + '/accuracy', accuracy)
        accuracy = float((fn + tp) / (fp + tp + fn + tn))
    return accuracy

# 用网络测试数据，训练出最好的阈值，以及阈值对应的测试准确率
# def test_rate(image_l,image_r,label_file):
#     # 100个0-1的阈值点生成
#     list_ = []
#     with open(label_file, 'r') as f:
#         for i in f:
#             i = i.strip()
#             i = i.replace('\n', ' ')
#             i = i.split(" ")
#             list_.append(i)
#     threshlod = wrt.produce_threshold()
#     # 1000张图片采用十折交叉法分成9:1,且迭代10次来训练出阈值
#     accuracy_ten = []
#     threshlod_ten = []
#     separate_image = random.sample(range(0, 1001), 1000)
#     kf = KFold(n_splits=10)
#     for train, test in kf.split(separate_image):
#         th_ac_tra_dic = {}  # 将100个阈值点，对应的训练准确率存到字典中
#         for tr_ld in threshlod:
#             num_e,num_r,TP,TN,FP,FN = 0,0,0,0,0,0
#             for num_tra in train:
#                 #print('第',num,'次的tain结果：\n')
#                 num_r += 1
#                 pic_tra_l,pic_tra_r = wrt.get_one_image(image_l, image_r, num_tra)
#                 # 对得到的图片做预处理：
#                 pic_tra_l = wrt.prepare(pic_tra_l[0])
#                 pic_tra_r = wrt.prepare(pic_tra_r[0])
#                 tes_tra_l = siamese(pic_tra_l)
#                 tes_tra_r = siamese(pic_tra_r)
#                 #tes_tra_loss为张量，转换为array值运算
#                 tes_tra_loss, tes_tra_distace = loss(tes_tra_l, tes_tra_r, int(list_[num_tra][0]))
#                 #将阈值np格式转换成tensor类型与tes_tra_distace作对比
#                 bol_tra = match(tes_tra_distace, tf.convert_to_tensor(tr_ld))
#                 if bol_tra == 0 and int(list_[num_tra][0]) == 0:
#                     TP += 1
#                 if bol_tra == 1 and int(list_[num_tra][0]) == 0:
#                     TN += 1
#                 if bol_tra == 0 and int(list_[num_tra][0]) == 1:
#                     FP += 1
#                 if bol_tra == 1 and int(list_[num_tra][0]) == 1:
#                     FN += 1
#                 if num_r % 20 == 0:
#                     ac_tra = evaluation(FP, TP, TN, FN)
#                     print('第',num_r,'次--训练集的训练率：', ac_tra)
#                     #th_ac_tra_dic[tr_ld] = ac_tra
#                 if num_r == len(train)+1 :
#                     ac_tra = evaluation(FP, TP, TN, FN)
#                     th_ac_tra_dic[tr_ld] = ac_tra
#                 else:continue
#             # ac_tra = evaluation(FP, TP, TN, FN)
#             # print('----测试集的准确率：',ac_tra)
#             # th_ac_tra_dic[tr_ld] = ac_tra
#             perfect_thre = max(th_ac_tra_dic, key=th_ac_tra_dic.get)
#             threshlod_ten.append(float(perfect_thre))
#             # accuracy_ten.append(th_ac_dic[perfect_thre])
#             for num_tes in test:
#                 #print('第', num, '次的test结果：\n')
#                 num_e += 1
#                 pic_tes = wrt.get_one_image(image_l, image_r, num_tes)
#                 pic_tes_l = wrt.prepare(pic_tes[0])
#                 pic_tes_r = wrt.prepare(pic_tes[1])
#                 tes_tes_l = siamese(pic_tes_l)
#                 tes_tes_r = siamese(pic_tes_r)
#                 tes_tes_loss, tes_tes_distance = loss(tes_tes_l, tes_tes_r, int(list_[num_tes][0]))
#                 bol_tes = match(tes_tes_distance, tf.convert_to_tensor(perfect_thre))
#                 if bol_tes == 0 and list_[num_tes] == 0:
#                     TP += 1
#                 if bol_tes == 1 and list_[num_tes] == 0:
#                     TN += 1
#                 if bol_tes == 0 and list_[num_tes] == 1:
#                     FP += 1
#                 if bol_tes == 1 and list_[num_tes] == 1:
#                     FN += 1
#                 if num_e % 20 ==0:
#                     ac_tes = evaluation(FP, TP, TN, FN)
#                     print('第',num_e,'次--训练集的测试率：', ac_tes)
#                     #accuracy_ten.append(float(ac_tes))
#                 else:continue
#             # ac_tes = evaluation(FP, TP, TN, FN)
#             # print('----训练集的准确率：', ac_tes)
#             accuracy_ten.append(float(np.mean(ac_tes)))
#     best_thrshold = threshlod_ten[accuracy_ten.index((max(accuracy_ten)))]
#     best_accuracy = np.mean(accuracy_ten)
#     return best_thrshold ,best_accuracy
