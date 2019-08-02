from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from scipy import signal
import cv2

slim = tf.contrib.slim
# trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
tf.reset_default_graph()

def fire_module(inputs, squeeze_depth, expand_depth, reuse=None, scope=None, outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)
            return outputs

def squeeze(inputs, num_outputs):
    return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')

def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat([e1x1, e3x3], 3)

def siamese_loss(out1,out2,y,Q=5):
    Q = tf.constant(Q, name="Q",dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1-out2) + 1e-10))
    pos = tf.multiply(tf.multiply(1 - y, 2 / Q), E_w)
    neg = tf.multiply(tf.multiply(y,2*Q),tf.exp(-2.77*E_w/Q))
    loss = pos + neg
    loss = tf.reduce_mean(loss)
    return loss ,E_w

def distance(out1, out2):
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1 - out2) + 1e-10))
    dit = tf.reduce_mean(E_w)
    return dit

def robin(img):
    im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    list_edge = []
    k1 = np.array([[1,1,1],[1,-2,1],[-1,-1,-1]])
    img_k1 = signal.convolve2d(im, k1, mode='same', boundary='fill', fillvalue=0)
    list_edge.append(np.abs(img_k1))
    k2 = np.array([[1,1,1],[-1,-2,-1],[-1,-1,1]])
    img_k2 = signal.convolve2d(im, k2, mode='same', boundary='fill', fillvalue=0)
    list_edge.append(np.abs(img_k2))
    k3 = np.array([[-1,1,1],[-1,-2,1],[-1,1,1]])
    img_k3 = signal.convolve2d(im, k3, mode='same', boundary='fill', fillvalue=0)
    list_edge.append(np.abs(img_k3))
    k4 = np.array([[-1,-1,1],[-1,-2,1],[1,1,1]])
    img_k4 = signal.convolve2d(im, k4, mode='same', boundary='fill', fillvalue=0)
    list_edge.append(np.abs(img_k4))
    k5 = np.array([[-1,-1,-1],[1,-2,1],[1,1,1]])
    img_k5 = signal.convolve2d(im, k5, mode='same', boundary='fill', fillvalue=0)
    list_edge.append(np.abs(img_k5))
    k6 = np.array([[1,-1,-1],[1,-2,-1],[1,1,1]])
    img_k6 = signal.convolve2d(im, k6, mode='same', boundary='fill', fillvalue=0)
    list_edge.append(np.abs(img_k6))
    k7 = np.array([[1,1,-1],[1,-2,-1],[1,1,-1]])
    img_k7 = signal.convolve2d(im, k7, mode='same', boundary='fill', fillvalue=0)
    list_edge.append(np.abs(img_k7))
    k8 = np.array([[1,1,1],[1,-2,-1],[1,-1,-1]])
    img_k8 = signal.convolve2d(im, k8, mode='same', boundary='fill', fillvalue=0)
    list_edge.append(np.abs(img_k8))
    edge = list_edge[0]
    for i in range(len(list_edge)):
        edge = edge*(edge>=list_edge[i]) + list_edge[i]*(edge<list_edge[i])
    edge[edge > 255] = 255
    edge = edge.astype(np.uint8)
    return edge

def match(disatance,dela):
    #dela阈值，距离小于阈值判断为匹配，距离大于阈值判断为不匹配，0代表匹配，1代表不匹
    if disatance >= dela:
        boo = 0
    else:boo=1
    return boo
#my_net
def siamese(inputs):
    net = slim.conv2d(inputs, 16, [3, 3], stride=2, padding='VALID', scope='conv1')
    net = slim.conv2d(net, 32, [3, 3], stride=2, padding='VALID', scope='conv2')
    net = slim.conv2d(net, 64, [3, 3], stride=1, padding='VALID', scope='conv3')
    net = slim.max_pool2d(net,  [3,3], stride=2, padding='VALID',scope='max_pool1')
    net = slim.conv2d(net, 80, [3, 3], stride=2, padding='VALID', scope='conv4')
    net = slim.conv2d(net, 80, [3, 3], stride=2, padding='VALID', scope='conv5')
    net = slim.conv2d(net, 100, [3, 3], stride=1, padding='VALID', scope='conv6')
    net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='max_pool2')
    return net
#inception_v3 net
def inception_v3(inputs):
    depth = lambda d: max(int(d * 1), 16)
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
        net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope='Conv2d_1a_3x3')
        net = slim.conv2d(net, depth(32), [3, 3], scope='Conv2d_2a_3x3')
        net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
        net = slim.conv2d(net, depth(80), [1, 1], scope='Conv2d_3b_1x1')
        net = slim.conv2d(net, depth(192), [3, 3], scope='Conv2d_4a_3x3')
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')
    # Inception blocks Inception核心模块
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):#设置补零
        with tf.variable_scope('Mixed_5b'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(32), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        with tf.variable_scope('Mixed_5c'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
                branch_1 = slim.conv2d(branch_1, depth(64), [5, 5], scope='Conv_1_0c_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        with tf.variable_scope('Mixed_5d'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        with tf.variable_scope('Mixed_6a'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
            #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_6a')
            #print_actication(net)
        with tf.variable_scope('Mixed_6b'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(128), [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(128), [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, depth(128), [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, depth(128), [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_6b')
            #print_actication(net)
        with tf.variable_scope('Mixed_6c'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_6c')
            #print_actication(net)
        with tf.variable_scope('Mixed_6d'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_6d')
            #print_actication(net)
        with tf.variable_scope('Mixed_6e'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(192), [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, depth(192), [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_6e')
            #print_actication(net)
        with tf.variable_scope('Mixed_7a'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
            #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_7a')
            #print_actication(net)
        with tf.variable_scope('Mixed_7b'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = tf.concat(axis=3, values=[
                    slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                    slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(
                    branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = tf.concat(axis=3, values=[
                    slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                    slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(
                    branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_7b')
            #print_actication(net)
        with tf.variable_scope('Mixed_7c'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = tf.concat(axis=3, values=[
                    slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                    slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(
                    branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = tf.concat(axis=3, values=[
                    slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                    slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(
                    branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            #net = tf.layers.batch_normalization(net, training=Train, name='bn_mixed_7c')
            #print_actication(net)
    return net
#mobile_v2 net
def mobilenet_v2(inputs, width_multiplier=1, is_training=True, scope='MobileNet'):
    def _depthwise_separable_conv(inputs, num_pwc_filters, width_multiplier, sc, downsample=False):
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=_stride,
                                                      depth_multiplier=1,
                                                      kernel_size=[3, 3],
                                                      scope=sc + '/depthwise_conv')
        bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return bn

    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                activation_fn=tf.nn.relu,
                                fused=True):
                net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME',
                                         scope='conv_1')
                net = slim.batch_norm(net, scope='conv_1/batch_norm')
                net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
                net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
                net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
                net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')

                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')

                net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
                net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
                net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')
    return net
#sequeezenet
def sequeeze(inputs, phase_train=True, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with tf.variable_scope('squeezenet', [inputs], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                net = slim.conv2d(inputs, 96, [7, 7], stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
                net = fire_module(net, 16, 64, scope='fire2')
                net = fire_module(net, 16, 64, scope='fire3')
                net = fire_module(net, 32, 128, scope='fire4')
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='maxpool4')
                net = fire_module(net, 32, 128, scope='fire5')
                net = fire_module(net, 48, 192, scope='fire6')
                net = fire_module(net, 48, 192, scope='fire7')
                net = fire_module(net, 64, 256, scope='fire8')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
                net = fire_module(net, 64, 256, scope='fire9')
                net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv10')
                net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avgpool10')
                net = tf.squeeze(net, [1, 2], name='logits')
    return net
#获得标准差
def junzhi(in1,in2):
    out = tf.subtract(in1,in2)
    x_mean = tf.reduce_mean(out)
    var = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.subtract(out,x_mean))))
    return var
#标准差对比
def new_loss(input1, input2, input3, Q=8):
    out1 = junzhi(input1,input2)
    out2 = junzhi(input2,input3)
    loss = out1 - Q * out2
    return loss


