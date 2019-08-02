import tensorflow as tf
import sia_cnn as si
import argparse
from sklearn.model_selection import KFold
import random
import write_read_tfrecords as wrt
import os
import numpy as np
import matplotlib.pyplot as plt
import time

batch_size = 1

img_l_tra = './CACD_data/train/train_l'
img_r_tra = './CACD_data/train/train_r'
label_train = './CACD_data/train/train.txt'

parser = argparse.ArgumentParser()
args = parser.parse_args()
parser.add_argument('--gpu', default=' ', type=str)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

saver = tf.train.import_meta_graph('./CACD_data/test/mt_sia_model_1/test_model.ckpt-59999.meta')
graph = tf.get_default_graph()
xl = graph.get_tensor_by_name('xl:0')
xr = graph.get_tensor_by_name('xr:0')
distance = tf.get_collection('pic_distance')[0]

with tf.Session() as sess:
    start = time.time()
    saver.restore(sess,'./CACD_data/test/mt_sia_model_1/test_model.ckpt-59999')
    list_ = []
    with open(label_train, 'r') as f:
        for i in f:
            i = i.strip()
            i = i.replace('\n', '')
            i = i.split(" ")
            list_.append(i)
    picture_l = np.ones(shape=[batch_size, 225, 225, 3])
    picture_r = np.ones(shape=[batch_size, 225, 225, 3])
    threshold = wrt.produce_threshold()
    accuracy_ten ,threshold_ten = [], []
    TPR, FPR = [], []
    get_num = random.sample(range(0, 3001), 3000)
    kf = KFold(n_splits=5)
    for tra_num, tes_num in kf.split(get_num):
        th_ac_tra_dic = {}
        for ld in threshold:
            num_r, tes_tra_ture = 1, 0
            print('------在阈值：%f 下的训练数据：------' % (ld))
            acc_tra = []
            for num_tra in tra_num:
                pic_tra = wrt.get_pic(img_l_tra, img_r_tra, num_tra)
                picture_l[0] = pic_tra[0]
                picture_r[0] = pic_tra[1]
                tra_distance = sess.run(distance,feed_dict={xl: picture_l, xr: picture_r})
                tra_bol = si.match(tra_distance, ld)
                # print('训练集第%d次，图片%d的距离值%f' %(num_r,num_tra, tra_distance), '\t图片的标签为：', list_[num_tra][0], '\t测试的标签为：', tra_bol)
                if (tra_bol == int(list_[num_tra][0])):
                    tes_tra_ture += 1
                if num_r % 20 == 0:
                    ac_tra = float(tes_tra_ture / 20)
                    print('第', num_r, '次--训练的训练率：', ac_tra)
                    acc_tra.append(ac_tra)
                    tes_tra_ture = 0
                num_r += 1
                if (num_r - 1) == len(tra_num):
                    th_ac_tra_dic[ld] = np.mean(acc_tra)
                else:
                    continue
        perfect_ld = max(th_ac_tra_dic, key=th_ac_tra_dic.get)
        threshold_ten.append(float(perfect_ld))
        # num_t, tes_tes_ture = 1, 0
        num_t = 1
        acc_tes = []
        tp, tn, fn, fp,tp1, tn1, fn1, fp1 = 0, 0, 0, 0, 0, 0, 0, 0
        print('\t在%d阈值下的测试：'%(perfect_ld))
        for num_tes in tes_num:
            pic_tes = wrt.get_pic(img_l_tra, img_r_tra, num_tes)
            picture_l[0] = pic_tes[0]
            picture_r[0] = pic_tes[1]
            tes_distance = sess.run(distance, feed_dict={xl: picture_l, xr: picture_r})
            tes_bol = si.match(tes_distance, perfect_ld)
            print('测试集第%d次，图片%d的距离值%f' %(num_t,num_tes, tes_distance), '\t图片的标签为：', list_[num_tes][0], '\t测试的标签为：', tes_bol)
            if (tes_bol == 0 and list_[num_tes][0]=='0'):
                tp += 1
                tp1 += 1
            if (tes_bol == 0 and list_[num_tes][0]=='1'):
                tn += 1
                tn1 += 1
            if (tes_bol == 1 and list_[num_tes][0]=='0'):
                fp += 1
                fp1 += 1
            if (tes_bol == 1 and list_[num_tes][0]=='1'):
                fn += 1
                fn1 += 1
            if num_t % 20 == 0:
                ac_tes = (tp + tn)/(tp + tn + fp + fn )
                print('第', num_t, '次--测试的测试率：', ac_tes)
                tp, tn, fn, fp = 0, 0, 0, 0
                acc_tes.append(ac_tes)
                # tes_tes_ture = 0
            num_t += 1
            if (num_t - 1) == len(tes_num):
                print('测试集的测试率为：', np.mean(acc_tes))
                accuracy_ten.append(np.mean(acc_tes))
            else:
                continue

        FPR.append((fp1)/(fp1+tn1))
        TPR.append((tp1)/(tp1+fn1))
    # print('5个最优的阈值为：', threshold_ten, '\t其准确率为：', accuracy_ten)
    # # best_threshold = threshold_ten[accuracy_ten.index(max(accuracy_ten))]
    print(FPR, TPR)
    best_accuracy = np.mean(accuracy_ten)
    print('其准确率：', best_accuracy)
    print('use time :',str(time.time()-start))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(FPR,TPR,color='black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Reciver Operating Characteristics')
    plt.savefig('./CACD_data/train/load/loss_3')
print('test finish')