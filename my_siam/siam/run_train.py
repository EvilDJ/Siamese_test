import siamese_net as sn
import os
import numpy as np
import tensorflow as tf
import write_read_tfrecords as wrt
import datetime

Batch_size = 20
Max_step = 10000

Log_train_dir = "./train_data/"
Records_file_l = "E:/PY_代码/New_Siamese/CACD_data/test/tes_l_new.tfrecords"
Records_file_r = "E:/PY_代码/New_Siamese/CACD_data/test/tes_r_new.tfrecords"
#训练集，record文件读取数据
tra_bt_l, tra_labt_l = wrt.read_records(Records_file_l, Batch_size)
tra_bt_l = tf.cast(tra_bt_l,dtype=tf.float32)
tra_labt_l = tf.cast(tra_labt_l,dtype=tf.float32)
tra_bt_r, tra_labt_r = wrt.read_records(Records_file_r, Batch_size)
tra_bt_r = tf.cast(tra_bt_r,dtype=tf.float32)
tra_labt_r = tf.cast(tra_labt_r,dtype=tf.float32)
# #训练的数据传输网络内
with tf.variable_scope('high_feature') as scope:
    high_feature_l = sn.siamese(tra_bt_l)
    # scope.reuse_variables()
    high_feature_r = sn.siamese(tra_bt_r)
y_l = np.argmax(tra_labt_l, axis=0)
#-------------学习率的设置以及反向计算的设置------------
global_step = tf.Variable(0)
train_loss , distance_tra = sn.loss(high_feature_l,high_feature_r,y_l)
Learn_rate = tf.train.exponential_decay(0.01,global_step,decay_steps=Max_step/Batch_size,decay_rate=0.98,staircase=True)
optimizer = tf.train.AdamOptimizer(Learn_rate).minimize(train_loss)
'''
   测试的数据传输到网络内，测试集数据按照十折交叉法，将测试数据分10份，
   取9份作为训练阈值，一份对阈值做评判。10次十折交叉，得到每次的准确率，求其平均值。
   得到，最优准确率对应的阈值，和网络的平均准确率
'''
#---------Tensorflow-----会话框执行前语句设置------
tf.summary.scalar('train_loss',train_loss)
# tf.summary.scalar('test_accuracy',accuracy)
summary_op = tf.summary.merge_all()
# optimizer = tf.train.AdamOptimizer(learning_rate=Learn_rate).minimize(train_loss)
init = tf.global_variables_initializer()
#-----Tensorflow----会话框-----P
print('run started!')
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(Log_train_dir, sess.graph)
    saver = tf.train.Saver()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for step in np.arange(Max_step):
            if coord.should_stop():
                break
            tra_loss = sess.run(train_loss)
            learn_rate = sess.run(Learn_rate)
            #perfe_thre = sess.run(perfect_threshold)
            if step % 100 == 0:
                sess.run(optimizer)
                print('Step %d, learn_rate = 0.7%f, train loss = %.7f\n' % (step,learn_rate, tra_loss))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            if step % 2000 == 0 or (step + 1) == Max_step:
                checkpoint_path = os.path.join(Log_train_dir, 'train_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
print('run finished! ')