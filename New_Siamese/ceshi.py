import numpy as np
from sklearn.model_selection import KFold
import random
import os
import cv2
import operator

# for i in range(1000):
#     print((i+1)%20)

# sep = random.sample(range(0,3001),3000)
# kf = KFold(n_splits=5)
# for i,j in kf.split(sep):
#     print(i)
#     print(j)

# label_train = './CACD_data/train/train.txt'
# list_ = []
# with open(label_train,'r') as f:
#     for i in f:
#         i = i.strip()
#         i = i.replace('\n','')
#         i = i.split( " ")
#         list_.append(i)
# print(len(list_))
# for i in list_:
#     print(i)

img_l_tra = './CACD_data/train/train_l'
img_r_tra = './CACD_data/train/train_r'


def get_pic(image_flie_l, image_flie_r, num):
    dictionary_l = {}
    for i in os.listdir(image_flie_l):  # listdir的参数是文件夹的路径
        value_num = int(i[6:-4])
        dictionary_l[i] = value_num
    list_l = sorted(dictionary_l.items(), key=operator.itemgetter(1))
    fil_l = image_flie_l + '/' + list_l[num][0]
    print(fil_l)
    image_l = cv2.imread(fil_l)
    print(type(image_l))#<class 'numpy.ndarray'>
    img_l = cv2.resize(image_l, (225, 225))
    # img_l= to_tensor(image_l)

    dictionary_r = {}
    for i in os.listdir(image_flie_r):  # listdir的参数是文件夹的路径
        value_num = int(i[6:-4])
        dictionary_r[i] = value_num
    list_r = sorted(dictionary_r.items(), key=operator.itemgetter(1))
    fil_r = image_flie_r + '/' + list_r[num][0]
    print(fil_r)
    image_r = cv2.imread(fil_r)
    print(type(image_r))
    img_r = cv2.resize(image_r, (225, 225))
    # img_r = to_tensor(image_r)
    return img_l, img_r
    # img_l = np.ones(shape=[])
    # img_r = np.ones(shape=[])
    # for i in os.listdir(image_flie_l):  # listdir的参数是文件夹的路径
    #     if str(num) in i:
    #         fil_l = image_flie_l + '/' + i
    #         print(fil_l)
    #         image_l = cv2.imread(fil_l)
    #         print(type(image_l))
    #         img_l = cv2.resize(image_l, (225, 225))
    #     else:continue
    #
    # for i in os.listdir(image_flie_r):
    #     if str(num) in i:
    #         fil_r = image_flie_r + '/' + i
    #         print(fil_r)
    #         image_r = cv2.imread(fil_r)
    #         print(type(image_r))
    #         img_r = cv2.resize(image_r, (225, 225))
    #     else:continue

get_pic(img_l_tra,img_r_tra,3)