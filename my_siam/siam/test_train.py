import siamese_net as sn
import tensorflow as tf
import os
import write_read_tfrecords as wrt
import numpy as np

Batch_size = 20
Max_step = 20000


# image_l = "./CACD_data/test/test_l/"
# image_r = "./CACD_data/test/test_r/"
# label_file = './CACD_data/test/test.txt'
# Log_train_dir = './CACD_data/test/'


Log_train_dir = "./CACD_data/test/load"
Records_file_l = './CACD_data/test/tes_l.tfrecords'
Records_file_r = './CACD_data/test/tes_r.tfrecords'
#训练集，record文件读取数据
tra_bt_l, tra_labt_l = wrt.read_records(Records_file_l, Batch_size)
tra_bt_l = tf.cast(tra_bt_l,dtype=tf.float32)
tra_labt_l = tf.cast(tra_labt_l,dtype=tf.float32)
tra_bt_r, tra_labt_r = wrt.read_records(Records_file_r, Batch_size)
tra_bt_r = tf.cast(tra_bt_r,dtype=tf.float32)
# #训练的数据传输网络内
high_feature_l= sn.siamese(tra_bt_l)
high_feature_r= sn.siamese(tra_bt_r)
y_l = np.argmax(tra_labt_l,axis=0)
train_loss , distance_tra = sn.loss(high_feature_l,high_feature_r,float(y_l))
# list_ = []
# with open(label_file, 'r') as f:
#     for i in f:
#         i = i.strip()
#         i = i.replace('\n', ' ')
#         i = i.split(" ")
#         list_.append(i)
# for i in range(len(list_)):
#     pic_tra_l, pic_tra_r = wrt.get_one_image(image_l, image_r, i)
#     pic_tra_l = wrt.prepare(pic_tra_l[0])
#     pic_tra_r = wrt.prepare(pic_tra_r[0])
#     tes_tra_l = sn.siamese(pic_tra_l)
#     tes_tra_r = sn.siamese(pic_tra_r)
#     tes_tra_loss, tes_tra_distace = sn.loss(tes_tra_l, tes_tra_r, float(list_[i][0]))

global_ = tf.Variable(0)
Learn_rate = tf.train.exponential_decay(0.01, global_, decay_steps=100, decay_rate=0.78, staircase=True)
optimizer = tf.train.AdamOptimizer(Learn_rate).minimize(train_loss)
# ---------Tensorflow-----会话框执行前语句设置------
tf.summary.scalar('train_loss', train_loss)
summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(Log_train_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    dist = []
    tra_loss = sess.run(train_loss)
    learn_rate = sess.run(Learn_rate)
    distance = sess.run(distance_tra)
    print('distance:',distance)
    #dist.append(distance)
    for i in range(Max_step):
        if i % 100 == 0:
            sess.run(optimizer)
            print('Step %d, learn_rate = %f, train loss = %f, pic distance= %f\n' % (i, learn_rate, tra_loss, distance))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, i)
        if i % 3000 == 0 or (i+1) == Max_step:
            checkpoint_path = os.path.join(Log_train_dir, 'train_model.ckpt')
            saver.save(sess, checkpoint_path, global_step=i)
print('run finished! ')

