from datetime import datetime as dt
import time
import math
import tensorflow as tf
import spp_layer

batch_size = 32
num_batches = 100
def print_actication(t):
    print((t.op.name),' ',t.get_shape().as_list())
def inference(images):
    parameters = []
    with tf.name_scope('conv1') as scope:
        kernel =tf.Variable(tf.truncated_normal([11,11,3,64],
                            dtype=tf.float32,stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_actication(conv1)
        parameters += [kernel,biases]
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        print_actication(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,64,192],
                                                 dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print_actication(conv2)
        parameters += [kernel, biases]
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        print_actication(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,192,384],
                                                 dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print_actication(conv3)
        parameters += [kernel, biases]
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
        print_actication(pool3)
    print("+++++-----")
    bins = 3
    #map_size = conv3.get_shape().as_list()[2]  # 14
    print(conv3.get_shape())
    print('-----++++++')
    c3_pool = spp_layer.spp(conv3,bins)
    #sppool = sppLayer.spatial_pyramid_pooling(conv3)
    print('+-+-+-+-+-+')
    #num = sppool.get_shape().as_list()[1]
    print_actication(c3_pool)
    print(c3_pool.get_shape()[1])
    return pool3 , parameters
def time_tensorflow_run(session ,target, info_string):
    num_steps_burn_in =10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches+num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i>=num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f'%(dt.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %(dt.now(), info_string, num_batches, mn, sd))


with tf.Graph().as_default():
    image_size = 224
    images = tf.Variable(tf.random_normal([batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))

    #img = cv2.imread('wx.jpg')
    pool3, parameters = inference(images)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    time_tensorflow_run(sess, pool3, "Forward")
    objecttive = tf.nn.l2_loss(pool3)
    grad = tf.gradients(objecttive, parameters)
    time_tensorflow_run(sess, grad, "Forward_backward")

'''
 with tf.name_scope('fc1') as scope:
        x_flat = tf.reshape(pool3, shape=[-1, 7 * 7 * 128])
        w_fc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 128, 1024], stddev=0.05, mean=0), name='w_fc1')
        b_fc1 = tf.Variable(tf.zeros(1024), name='b_fc1')
        fc1 = tf.add(tf.matmul(x_flat, w_fc1), b_fc1)
    with tf.name_scope('relu_fc1') as scope:
        relu_fc1 = tf.nn.relu(fc1, name='relu_fc1')
    with tf.name_scope('drop_1') as scope:
        drop_1 = tf.nn.dropout(relu_fc1, keep_prob=keep_prob, name='drop_1')
    with tf.name_scope('bn_fc1') as scope:
        bn_fc1 = tf.layers.batch_normalization(drop_1, name='bn_fc1')

    with tf.name_scope('fc2') as scope:
        w_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 512], stddev=0.05, mean=0), name='w_fc2')
        b_fc2 = tf.Variable(tf.zeros(512), name='b_fc2')
        fc2 = tf.add(tf.matmul(bn_fc1, w_fc2), b_fc2)
    with tf.name_scope('relu_fc2') as scope:
        relu_fc2 = tf.nn.relu(fc2, name='relu_fc2')
    with tf.name_scope('drop_2') as scope:
        drop_2 = tf.nn.dropout(relu_fc2, keep_prob=keep_prob, name='drop_2')
    with tf.name_scope('bn_fc2') as scope:
        bn_fc2 = tf.layers.batch_normalization(drop_2, name='bn_fc2')

    with tf.name_scope('fc3') as scope:
        w_fc3 = tf.Variable(tf.truncated_normal(shape=[512, 2], stddev=0.05, mean=0), name='w_fc3')
        b_fc3 = tf.Variable(tf.zeros(2), name='b_fc3')
        fc3 = tf.add(tf.matmul(bn_fc2, w_fc3), b_fc3)
'''
   