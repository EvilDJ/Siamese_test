import siamese_net as sn
import tensorflow as tf
from sklearn.model_selection import KFold
import random
import write_read_tfrecords as wrt
import numpy as np

image_l = "./test_data/test_l/tes_l/"
image_r = "./test_data/test_r/tes_r/"
label_file = './test_data/test_l/tes_l.txt'
Log_test_dir = './test_data/'
log_train_dir = './train_data/'

xl_test = tf.placeholder(tf.float32,shape=(1,225,225,3))
xr_test = tf.placeholder(tf.float32,shape=(1,225,225,3))
label = tf.placeholder(tf.float32)

tes_tra_l = sn.siamese(xl_test)
tes_tra_r = sn.siamese(xr_test)
tes_tra_loss, tes_tra_distace = sn.loss(tes_tra_l, tes_tra_r, label)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print('run test start')
    # print("Reading checkpoints...")
    # ckpt = tf.train.get_checkpoint_state(log_train_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     print('Loading success, global_step is %s' % global_step)
    # else:
    #     print('No checkpoint file found')
    
    list_ = []
    with open(label_file, 'r') as f:
        for i in f:
            i = i.strip()
            i = i.replace('\n', ' ')
            i = i.split(" ")
            list_.append(i)
    # 100个0-1的阈值点生成
    threshlod = wrt.produce_threshold()
    # 1000张图片采用十折交叉法分成9:1,且迭代10次来训练出阈值
    accuracy_ten = []
    threshlod_ten = []
    separate_image = random.sample(range(0, 1001), 1000)
    kf = KFold(n_splits=10)
    for train, test in kf.split(separate_image):
        th_ac_tra_dic = {}  # 将100个阈值点，对应的训练准确率存到字典中
        for tr_ld in threshlod:
            num_r, tes_ture = 1, 0
            print('在阈值：%f 下训练数据开始！-----' % (tr_ld))
            acc_tra = []
            for num_tra in train:
                # print('第',num,'次的tain结果：\n')
                pic_tra_l, pic_tra_r = wrt.get_one_image(image_l, image_r, num_tra)
                # 对得到的图片做预处理：
                pic_tra_l = wrt.prepare(pic_tra_l[0])
                #print(pic_tra_l)
                pic_tra_r = wrt.prepare(pic_tra_r[0])
                #print(list_[num_tra][0])
                pic_tra_l_s,pic_tra_r_s = sess.run([pic_tra_l,pic_tra_r])
                tes_tra_distace_s = sess.run(tes_tra_distace, feed_dict={xl_test:pic_tra_l_s,xr_test:pic_tra_r_s,label:int(list_[num_tra][0])})
                # 将阈值np格式转换成tensor类型与tes_tra_distace作对比
                print(tes_tra_distace_s,'\t',list_[num_tra][0],'\t',num_tra)  #数值型
                bol_tra = sn.match(tes_tra_distace_s, tr_ld)
                if (bol_tra == int(list_[num_tra][0])) :
                    tes_ture+=1
                if num_r % 20 == 0:
                    #ac_tra = sn.evaluation(FP, TP, TN, FN)
                    ac_tra = float(tes_ture/20)
                    print('第', num_r, '次--训练集的训练率：', ac_tra)  # ac_tra依旧是张量
                    acc_tra.append(ac_tra)
                num_r += 1
                if num_r == len(train):
                    th_ac_tra_dic[tr_ld] = np.mean(acc_tra)
                else:
                    continue
            # ac_tra = evaluation(FP, TP, TN, FN)
            # print('----测试集的准确率：',ac_tra)
            # th_ac_tra_dic[tr_ld] = ac_tra
        perfect_thre = max(th_ac_tra_dic, key=th_ac_tra_dic.get)
        threshlod_ten.append(float(perfect_thre))
        # accuracy_ten.append(th_ac_dic[perfect_thre])
        num_e, TP, TN, FP, FN = 0, 0, 0, 0, 0
        acc_tes = []
        for num_tes in test:
            # print('第', num, '次的test结果：\n')
            num_e += 1
            pic_tes = wrt.get_one_image(image_l, image_r, num_tes)
            pic_tes_l = wrt.prepare(pic_tes[0])
            pic_tes_r = wrt.prepare(pic_tes[1])
            pic_tes_l_s, pic_tes_r_s = sess.run([pic_tes_l, pic_tes_r])
            tes_tes_distance_s=sess.run(tes_tra_distace, feed_dict={xl_test:pic_tes_l_s,xr_test:pic_tes_r_s,label:int(list_[num_tes][0])})
            # tes_tes_l = sn.siamese(pic_tes_l)
            # tes_tes_r = sn.siamese(pic_tes_r)
            # tes_tes_loss, tes_tes_distance = sn.loss(tes_tes_l, tes_tes_r, int(list_[num_tes][0]))
            bol_tes = sn.match(tes_tes_distance_s, perfect_thre)
            if bol_tes == 0 and list_[num_tes] == 0:
                TP += 1
            if bol_tes == 1 and list_[num_tes] == 0:
                TN += 1
            if bol_tes == 0 and list_[num_tes] == 1:
                FP += 1
            if bol_tes == 1 and list_[num_tes] == 1:
                FN += 1
            if num_e % 20 == 0:
                ac_tes = sn.evaluation(FP, TP, TN, FN)
                print('第', num_e, '次--训练集的测试率：', ac_tes)
                acc_tes.append(ac_tes)
            if num_e == len(test):
                print('测试集的测试率：', np.mean(acc_tes))
                accuracy_ten.append(np.mean(acc_tes))
            else:
                continue
    print('十个最优阈值：', threshlod_ten)
    print('\n最优阈值对应的测试准确率：', accuracy_ten)
    best_thrshold = threshlod_ten[accuracy_ten.index((max(accuracy_ten)))]
    best_accuracy = np.mean(accuracy_ten)
    print('此网络的准确率：', best_accuracy)
    
print('test end')