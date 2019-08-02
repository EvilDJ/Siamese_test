#采用十折交叉法，9折为训练，1折为测试，训练出距离阈值
import tensorflow as tf
from sklearn.model_selection import KFold
import random
import numpy as np
import siamese
import write_read_tfrecords as wrt

img_l_tra = './CACD_data/train/train_l'
img_r_tra = './CACD_data/train/train_r'
label_train = './CACD_data/train/train.txt'

list_ = []
with open(label_train,'r') as f:
    for i in f:
        i = i.strip()
        i = i.replace('\n','')
        i = i.split( " ")
        list_.append(i)
threshold = wrt.produce_threshold()

with tf.Session() as sess:
    picture_l = np.zeros(shape=[1, 225, 225, 3], dtype=np.float32)
    picture_r = np.zeros(shape=[1, 225, 225, 3], dtype=np.float32)
    accuracy_ten = []
    threshold_ten = []
    get_num = random.sample(range(0,3001), 3000)
    kf = KFold(n_splits=10)
    for tra_num ,tes_num in kf.split(get_num):
        th_ac_tra_dic = {}
        for ld in threshold:
            num_r , tes_tra_ture = 1,0
            print('------在阈值：%f 下的训练数据：------'%(ld))
            acc_tra = []
            for num_tra in tra_num:
                pic_tra = wrt.get_pic(img_l_tra,img_r_tra,num_tra)
                picture_l[0] = pic_tra[0]
                picture_r[0] = pic_tra[1]
                tra_distance = siamese.siamese(picture_l,picture_r,sess)
                tra_bol = siamese.match(tra_distance,ld)
                print(num_tra,tra_distance,'\t图片的标签为：',list_[num_tra][0],'\t测试的标签为：',tra_bol)
                if (tra_bol == int(list_[num_tra][0])):
                    tes_tra_ture+=1
                if num_r % 20 == 0:
                    ac_tra = float(tes_tra_ture/20)
                    print('第',num_r,'次--训练的训练率：',ac_tra)
                    acc_tra.append(ac_tra)
                    tes_tra_ture = 0
                num_r += 1
                if (num_r-1) == len(tra_num):
                    th_ac_tra_dic[ld] = np.mean(acc_tra)
                else:continue
        perfect_ld = max(th_ac_tra_dic,key=th_ac_tra_dic.get)
        threshold_ten.append(float(perfect_ld))
        num_t , tes_tes_ture = 1, 0
        acc_tes = []
        for num_tes in tes_num:
            pic_tes = wrt.get_one_image(img_l_tra, img_r_tra, num_tes)
            picture_l[0] = pic_tes[0]
            picture_r[0] = pic_tes[1]
            tes_distance = siamese.siamese(picture_l, picture_r, sess)
            tes_bol = siamese.match(tes_distance, perfect_ld)
            print(num_tes,tes_distance, '\t图片的标签为：', list_[num_tes][0], '\t测试的标签为：', tes_bol)
            if (tes_bol == int(list_[tes_num][0])):
                tes_tes_ture += 1
            if num_t % 20 == 0:
                ac_tes = float(tes_tes_ture / 20)
                print('第', num_t, '次--训练的训练率：', ac_tes)
                acc_tes.append(ac_tes)
                tes_tes_ture = 0
            num_t += 1
            if (num_t - 1) == len(tes_num):
                print('测试集的测试率为：',np.mean(acc_tes))
                accuracy_ten.append(np.mean(acc_tes))
            else:
                continue
    print('10个最优的阈值为：',threshold_ten,'\n其准确率为：',accuracy_ten)
    best_threshold = threshold_ten[accuracy_ten.index(max(accuracy_ten))]
    best_accuracy = np.mean(accuracy_ten)
    print('最有的阈值：',best_threshold,'\n其准确率：',best_accuracy)
print('测试结束')