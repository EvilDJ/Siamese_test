import tensorflow as tf
import siamese as si
import numpy as np
import random
import write_read_tfrecords as wrt
import os

lr_init = 0.01
iterations = 20000
batch_size = 20

image_l_tes = "./CACD_data/test/test_l/"
image_r_tes = "./CACD_data/test/test_r/"
label_tes = "./CACD_data/test/test.txt"
Log_train_dir = "./CACD_data/test/load"
# image_l_tf = "./CACD_data/test/tes_l_new.tfrecords"
# image_r_tf = "./CACD_data/test/tes_r_new.tfrecords"

with tf.variable_scope('data') as scope:
    xl = tf.placeholder(tf.float32, shape=[batch_size, 225, 225, 3], name='xl')
    xr = tf.placeholder(tf.float32, shape=[batch_size, 225, 225, 3], name='xr')
    y = tf.placeholder(tf.float32, shape=[batch_size], name='y')

with tf.variable_scope('siamese') as scope:
    out1 = si.siamese(xl)
    scope.reuse_variables()
    out2 = si.siamese(xr)
    
with tf.variable_scope('metrics') as scope:
    loss,  dist = si.siamese_loss(out1, out2, y)
    tf.add_to_collection('pic_distance', dist)
    global_ = tf.Variable(tf.constant(0))
    lr = tf.train.exponential_decay(lr_init,global_,decay_steps=100,decay_rate=0.59,staircase=True)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

loss_summary = tf.summary.scalar('loss',loss)
merged_summary = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    m = 0
    
    writer = tf.summary.FileWriter(Log_train_dir,sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()
    
    picture_l = np.ones(shape=[20,225,225,3])
    picture_r = np.ones(shape=[20,225,225,3])
    label_ = np.ones(shape=[20])
    
    
    list_ = []
    with open(label_tes, 'r') as f:
        for i in f:
            i = i.strip()
            i = i.replace('\n', ' ')
            i = i.split(" ")
            list_.append(i)
    threshold = random.sample(range(0, len(list_)), len(list_))
    for fn in threshold:
        fn = int(fn)
        xs_l,xs_r = wrt.get_pic(image_l_tes,image_r_tes,fn)
        picture_l[m%20] = xs_l
        picture_r[m%20] = xs_r
        # label = list_[fn]
        label_[m%20] = np.array(list_[fn])
        if m % 20 ==0:
            # pic_l,pic_r = sess.run([xs_l,xs_r])#pic_l,pic_r 格式 naddary格式
            _,train_loss, summ, distance = sess.run([optimizer,loss, merged_summary, dist],
                                                              feed_dict={xl: picture_l, xr: picture_r, y: label_})
            # _, train_loss, summ, distance = sess.run([optimizer, loss, merged_summary, dist])
            print(m, list_[fn][0], '第%d次距离为：%f,loss : %f' % (fn, distance, train_loss))
            if m % 100 == 0:
                learn_rate = sess.run(lr, feed_dict={global_: m})
                # avg_los = sum(los) / 100
                # sess.run(optimizer)
                print('Step %d, learn_rate = %f, pic distance= %f\n' % (
                m, learn_rate, distance))
                writer.add_summary(summ, fn)
            if m % 4000 == 0 or (m + 1) == len(threshold):
                checkpoint_path = os.path.join(Log_train_dir, 'test_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=fn)
        
        m += 1
    
    
    
    # print('=-=--==-')
    # # xr_,xl_,y_ = sess.run([xr_tf,xl_tf,label_l])
    # # print('++6+6+6')
    # # train_loss, summ, distance = sess.run([loss, merged_summary, dist])
    # train_loss  = sess.run(loss)
    # print('66666666')
    # distance = sess.run(dist)
    # print('train_loss:',train_loss,'distance:',distance)
    # for i in range(iterations):
    #     print('+++++++')
    #     # print(labe,'第%d次距离为：%f,loss : %f' % (i, distance, train_loss))
    #     if i % 100 == 0:
    #         # learn_rate = sess.run(lr,feed_dict={global_:i})
    #         _,summ = sess.run([optimizer,merged_summary])
    #         # sess.run(optimizer)
    #         # print('Step %d, learn_rate = %f,  pic distance= %f\n' % (i, learn_rate, distance))
    #         print('Step %d,pic distance= %f\n' % (i,distance))
    #         writer.add_summary(summ,i)
    #         # los = []
    #         # print(los)
    #     if i % 4000 == 1 or (i + 1) == iterations:
    #         checkpoint_path = os.path.join(Log_train_dir, 'test_model.ckpt')
    #         saver.save(sess, checkpoint_path, global_step=i)
    #         break
    writer.close()