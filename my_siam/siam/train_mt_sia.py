import tensorflow as tf
import sia_cnn as si
import argparse
import random
import write_read_tfrecords as wrt
import os
import numpy as np
import matplotlib.pyplot as plt

lr_init = 0.01
batch_size = 1

image_l_tes = "./CACD_data/test/test_l/"
image_r_tes = "./CACD_data/test/test_r/"
label_tes = "./CACD_data/test/test.txt"
Log_train_dir = "./CACD_data/test/load"
loss_data = open(r'./CACD_data/test/load/loss_data','w')

parser = argparse.ArgumentParser()
args = parser.parse_args()
parser.add_argument('--gpu', default=' ', type=str)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

xl = tf.placeholder(tf.float32, [batch_size, 225, 225, 1], name='xl')
xr = tf.placeholder(tf.float32, [batch_size, 225, 225, 1], name='xr')
y = tf.placeholder(tf.float32, [batch_size], name='y')

with tf.variable_scope('siamese') as scope:
    out1 = si.siamese(xl)
    scope.reuse_variables()
    out2 = si.siamese(xr)
with tf.variable_scope('metrics') as scope:
    loss, dist = si.siamese_loss(out1, out2, y)
    tf.add_to_collection('pic_distance', dist)
    global_ = tf.Variable(tf.constant(0))
    lr = tf.train.exponential_decay(lr_init, global_, decay_steps=500, decay_rate=0.47, staircase=True)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

loss_summary = tf.summary.scalar('loss', loss)
merged_summary = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(Log_train_dir, sess.graph)
    m = 1
    picture_l = np.ones(shape=[batch_size, 225, 225, 1])
    picture_r = np.ones(shape=[batch_size, 225, 225, 1])
    label_ = np.ones(shape=[batch_size])

    list_ = []
    with open(label_tes, 'r') as f:
        for i in f:
            i = i.strip()
            i = i.replace('\n', ' ')
            i = i.split(" ")
            # print(i)
            list_.append(i)
    threshold = random.sample(range(0, len(list_)), len(list_))
    tra_loss = np.zeros([2 * len(threshold)])
    for i in range(2):
        for fn in threshold:
            fn = int(fn)
            xs_l, xs_r = wrt.get_pic(image_l_tes, image_r_tes, fn)
            # picture_l[m % batch_size] = xs_l
            # picture_r[m % batch_size] = xs_r
            label_[m % batch_size] = np.array(list_[fn])
            if m % batch_size == 0:
                _, learn_rate, train_loss, summ, distance = sess.run([optimizer, lr, loss, merged_summary, dist],
                                                                     feed_dict={xl: xs_l, xr: xs_r, y: label_,
                                                                                global_: m})
                print(m, list_[fn][0], 'learn_rate:', learn_rate, '第%d的距离为：' % (fn), distance, 'Loss:', train_loss)
                loss_data.write(str(train_loss) + '\n')
                tra_loss[m-1] = train_loss
                # print(los)
                if m % 100 == 0:
                    # print('Step %d, learn_rate = %f, pic distance= %f\n' % (m, learn_rate, distance))
                    writer.add_summary(summ, fn)
                if m % 6000 == 0 or m == 5 * len(threshold) - 1:
                    checkpoint_path = os.path.join(Log_train_dir, 'test_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=m)
            m += 1

    writer.close()
    loss_data.close()
fig , ax1 = plt.subplots()
ins1 = ax1.plot(np.arange(len(2 * threshold)), tra_loss, label='Loss', color='black')
ax1.set_xlabel('iiteration')
ax1.set_ylabel('traing loss')
labels = ['Loss']
plt.legend(ins1, labels, loc=7)
plt.savefig('./CACD_data/test/mt_sia_model_20/loss_pic_black')