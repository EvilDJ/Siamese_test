import siamese_net as sn
import tensorflow as tf
import write_read_tfrecords as wrt
import os
import cv2
#从box中筛选合适的方框格式boxes
def detective_box(box ,pic):# box = total_boxes
    box_match = []
    perfect_threshold = 3.1622778e-5 # 经过Siamese网络测试得到。
    
    modelpic_dir = './model_pic'
    xl_test = tf.placeholder(tf.float32, shape=(1, 225, 225, 3))
    xr_test = tf.placeholder(tf.float32, shape=(1, 225, 225, 3))
    tes_tra_l = sn.siamese(xl_test)
    tes_tra_r = sn.siamese(xr_test)
    tes_tra_distace = sn.dist(tes_tra_l, tes_tra_r)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for dir_n, dir_l, file_n in os.walk(modelpic_dir):
            for i in file_n:
                pat = dir_n + i
                img_model = wrt.prepare(pat)  # 获得model的图片格式
                for box_position in box:
                    face_position = box_position.astype(int)
                    crop = pic[face_position[1]:face_position[3], face_position[0]:face_position[2], ]
                    crop_b = cv2.resize(crop, (225, 225), interpolation=cv2.INTER_CUBIC)
                    crop_c = wrt.prepare(crop_b)
                    pic_tra_l_s, pic_tra_r_s = sess.run([img_model, crop_c])
                    tes_tra_distace_s = sess.run(tes_tra_distace, feed_dict={xl_test: pic_tra_l_s, xr_test: pic_tra_r_s})
                    # 判断两者是否相似
                    if tes_tra_distace_s <= perfect_threshold:
                        if box_position not in box_match:
                            box_match.append(box_position)
    
    return box_match

