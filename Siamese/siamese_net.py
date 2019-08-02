import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.reset_default_graph()
mnist = input_data.read_data_sets('./data/mnist',one_hot=True)

print(mnist.validation.num_examples)
print(mnist.train.num_examples)
print(mnist.test.num_examples)
def siamese_loss(out1,out2,y,Q=5):

    Q = tf.constant(Q, name="Q",dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1-out2),1))
    pos = tf.multiply(tf.multiply(y,2/Q),tf.square(E_w))
    neg = tf.multiply(tf.multiply(1-y,2*Q),tf.exp(-2.77/Q*E_w))
    loss = pos + neg
    loss = tf.reduce_mean(loss)
    return loss

def siamese(inputs,keep_prob):
        with tf.name_scope('conv1') as scope:
            w1 = tf.Variable(tf.truncated_normal(shape=[3,3,1,32],stddev=0.05),name='w1')
            b1 = tf.Variable(tf.zeros(32),name='b1')
            conv1 = tf.nn.conv2d(inputs,w1,strides=[1,1,1,1],padding='SAME',name='conv1')
        with tf.name_scope('relu1') as scope:
            relu1 = tf.nn.relu(tf.add(conv1,b1),name='relu1')
            
        with tf.name_scope('conv2') as scope:
            w2 = tf.Variable(tf.truncated_normal(shape=[3,3,32,64],stddev=0.05),name='w2')
            b2 = tf.Variable(tf.zeros(64),name='b2')
            conv2 = tf.nn.conv2d(relu1,w2,strides=[1,2,2,1],padding='SAME',name='conv2')
        with tf.name_scope('relu2') as scope:
            relu2 = tf.nn.relu(conv2+b2,name='relu2')
            
        with tf.name_scope('conv3') as scope:
            w3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,128],mean=0,stddev=0.05),name='w3')
            b3 = tf.Variable(tf.zeros(128),name='b3')
            conv3 = tf.nn.conv2d(relu2,w3,strides=[1,2,2,1],padding='SAME')
        with tf.name_scope('relu3') as scope:
            relu3 = tf.nn.relu(conv3+b3,name='relu3')
        #第一个全连接层
        with tf.name_scope('fc1') as scope:
            x_flat = tf.reshape(relu3,shape=[-1,7*7*128])
            w_fc1=tf.Variable(tf.truncated_normal(shape=[7*7*128,1024],stddev=0.05,mean=0),name='w_fc1')
            b_fc1 = tf.Variable(tf.zeros(1024),name='b_fc1')
            fc1 = tf.add(tf.matmul(x_flat,w_fc1),b_fc1)
        with tf.name_scope('relu_fc1') as scope:
            relu_fc1 = tf.nn.relu(fc1,name='relu_fc1')
        with tf.name_scope('drop_1') as scope:
            drop_1 = tf.nn.dropout(relu_fc1,keep_prob=keep_prob,name='drop_1')
        with tf.name_scope('bn_fc1') as scope:
            bn_fc1 = tf.layers.batch_normalization(drop_1,name='bn_fc1')
        #第二个全连接层
        with tf.name_scope('fc2') as scope:
            w_fc2 = tf.Variable(tf.truncated_normal(shape=[1024,512],stddev=0.05,mean=0),name='w_fc2')
            b_fc2 = tf.Variable(tf.zeros(512),name='b_fc2')
            fc2 = tf.add(tf.matmul(bn_fc1,w_fc2),b_fc2)
        with tf.name_scope('relu_fc2') as scope:
            relu_fc2 = tf.nn.relu(fc2,name='relu_fc2')
        with tf.name_scope('drop_2') as scope:
            drop_2 = tf.nn.dropout(relu_fc2,keep_prob=keep_prob,name='drop_2')
        with tf.name_scope('bn_fc2') as scope:
            bn_fc2 = tf.layers.batch_normalization(drop_2,name='bn_fc2')
        #第三个全连接层
        with tf.name_scope('fc3') as scope:
            w_fc3 = tf.Variable(tf.truncated_normal(shape=[512,2],stddev=0.05,mean=0),name='w_fc3')
            b_fc3 = tf.Variable(tf.zeros(2),name='b_fc3')
            fc3 = tf.add(tf.matmul(bn_fc2,w_fc3),b_fc3)
        return fc3
#超参数的设置
lr = 0.01
iterations = 20000
batch_size = 64
#给TensorFlow喂入数据：x1,x2,y,keep_prob
with tf.variable_scope('input_x1') as scope:
    x1 = tf.placeholder(tf.float32, shape=[None, 784])
    x_input_1 = tf.reshape(x1, [-1, 28, 28, 1])
with tf.variable_scope('input_x2') as scope:
    x2 = tf.placeholder(tf.float32, shape=[None, 784])
    x_input_2 = tf.reshape(x2, [-1, 28, 28, 1])
with tf.variable_scope('y') as scope:
    y = tf.placeholder(tf.float32, shape=[batch_size])
with tf.name_scope('keep_prob') as scope:
    keep_prob = tf.placeholder(tf.float32)
#对喂入的数据x1,x2传送的siamese网络中，得到loss值
with tf.variable_scope('siamese') as scope:
    out1 = siamese(x_input_1,keep_prob)
    scope.reuse_variables()
    out2 = siamese(x_input_2,keep_prob)
with tf.variable_scope('metrics') as scope:
    loss = siamese_loss(out1, out2, y)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

loss_summary = tf.summary.scalar('loss',loss)
merged_summary = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:

    writer = tf.summary.FileWriter('./graph/siamese',sess.graph)
    sess.run(tf.global_variables_initializer())
    #保存模型路径
    save_path = saver.save(sess,"./mynet/siamese_net.ckpt")
    print("Save to path:",save_path)

    for itera in range(iterations):
        xs_1, ys_1 = mnist.train.next_batch(batch_size)
        ys_1 = np.argmax(ys_1,axis=1)
        xs_2, ys_2 = mnist.train.next_batch(batch_size)
        ys_2 = np.argmax(ys_2,axis=1)
        y_s = np.array(ys_1==ys_2,dtype=np.float32)
        _,train_loss,summ = sess.run([optimizer,loss,merged_summary],feed_dict={x1:xs_1,x2:xs_2,y:y_s,keep_prob:0.6})

        writer.add_summary(train_loss,summ)
        if itera % 1000 == 1 :
            print('iter {},train loss {}'.format(itera,train_loss))
    embed = sess.run(out1,feed_dict={x1:mnist.test.images,keep_prob:0.6})
    test_img = mnist.test.images.reshape([-1,28,28,1])
    writer.close()