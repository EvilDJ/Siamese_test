# -*- coding:utf-8: -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    # print('y_pre=', np.shape(y_pre))
    # print('tf.argmax(y_pre, 1)=', sess.run(tf.argmax(y_pre, 1)))
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape, layer_name):
    with tf.name_scope('weights'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        weights = tf.Variable(initial, name='W')
        tf.summary.histogram(layer_name+'/weights', weights)
    return weights


def bias_variable(shape, layer_name):
    with tf.name_scope('biases'):
        initial = tf.constant(0.1, shape=shape)
        biases = tf.Variable(initial, name='b')
        tf.summary.histogram(layer_name+'biases', biases)
    return biases


def conv2d(x, W, layer_name):  # 卷积层函数
    # stride 格式为： [1, x_movement, y_movement, 1]
    # must have strides[0]=strides[3]=1
    outputs = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs
    # W: [filter_height, filter_width, in_channels, out_channels]


def max_pool_2x2(x, layer_name):  # 池化层函数
    outputs = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs


with tf.name_scope('inputs'):
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784], name='x_input')  # 28*28
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# keep_prob是保留概率，即我们要保留的结果所占比例，
# 它作为一个placeholder，在run时传入， 当keep_prob=1的时候，相当于100%保留，也就是dropout没有起作用。
x_image = tf.reshape(xs, [-1, 28, 28, 1], name='x_image')  # 图片高度为1
# print(x_image.shape)  # [n_samples, 28, 28, 1]

##########################################################################
###  构建整个卷积神经网络
##########################################################################

# conv1 layer #
with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32], 'conv1')  # patch 5*5, in_size=1. out_size=32
    b_conv1 = bias_variable([32], 'conv1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 'conv1')+b_conv1)  # output_size = 28*28*32

with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1, 'pool1')  # output_size = 14*14*32

with tf.name_scope('conv2'):
    # conv2 layer #
    W_conv2 = weight_variable([5, 5, 32, 64], 'conv2')  # patch 5*5, in_size=32. out_size=64
    b_conv2 = bias_variable([64], 'conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 'conv2')+b_conv2)  # output_size = 14*14*64

with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2, 'pool2')  # output_size = 7*7*64

with tf.name_scope('fc1'):
    # func1 layer #
    W_fc1 = weight_variable([7*7*64, 1024], 'fc1')
    b_fc1 = bias_variable([1024], 'fc1')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # [n_samples, 7, 7, 64]-->[n_samples,7*7*64]
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout防止过拟合

with tf.name_scope('fc2'):
    # func2 layer #
    W_fc2 = weight_variable([1024, 10], 'fc2')
    b_fc2 = bias_variable([10], 'fc2')
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)


with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('corss_entropy', cross_entropy)


with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # print batch_xs.shape, batch_ys.shape
    ## 输出 (100, 784) (100, 10)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        rs = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        writer.add_summary(rs, i)
        print(compute_accuracy(mnist.test.images[: 1000], mnist.test.labels[: 1000]))