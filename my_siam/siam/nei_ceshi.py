import os
import cv2
from skimage import io
import matplotlib.pyplot as plt
import random
from PIL import Image
from sklearn.model_selection import KFold
import write_read_tfrecords as wrt
import operator
import numpy as np
#from py_func_1 import *
import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops

#-----------------学习率测试----------
# learning_rate = 0.1
# decay_rate = 0.96
# global_steps = 5000
# decay_steps = 100
#
# global_ = tf.Variable(tf.constant(0))
# c = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True)
# d = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)
#
# T_C = []
# F_D = []
#
# with tf.Session() as sess:
#     for i in range(global_steps):
#         T_c = sess.run(c, feed_dict={global_: i})
#         T_C.append(T_c)
#         F_d = sess.run(d, feed_dict={global_: i})
#         F_D.append(F_d)
#
# plt.figure(1)
# plt.plot(range(global_steps), F_D, 'r-')
# plt.plot(range(global_steps), T_C, 'b-')
#
# plt.show()

# a=np.array([[5.,8.,2.],[7.,9.,1.]])
# a=np.expand_dims(a,axis=0)
#
# a=tf.constant(a,dtype=tf.float32)
# a_mean, a_var = tf.nn.moments(a, axes=[0,1],keep_dims=True)
# b=tf.rsqrt(a_var)
# c=(a-a_mean)*b
# d=tf.layers.batch_normalization(a,training=True)
# e=tf.nn.batch_normalization(a,a_mean,a_var,offset=None,scale=1,variance_epsilon=0)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# a_value,b_value,c_value,d_value,e_value=sess.run([a,b,c,d,e])
# print(d_value)
# print(e_value)
# sess.close()


# threshold = random.sample(range(0,1001),1000)
# kf = KFold(n_splits=10)
# num = 0
# for train, test in kf.split(threshold):
#     # tra_ = []
#     # tes_ = []
#     # for i in train:
#     #     tra_.append(threshold[i])
#     # for i in test:
#     #     tes_.append(threshold[i])
#     print("第",num,"轮\n","tain:",train)
#     print("test:",test)
#--------------------------------------------------
#     #print("%s %s" % (tra_, tes_),'\n')
#     num += 1
threshold = random.sample(range(0,100),100)
th = []
for i in threshold:
    th.append(float(int(i)/10000))
print(th)
# for i in threshold:
#     print(float(i/400))
#path = "./test_data/test_l/tes_l/"
# par = "./test_data/test_r/tes_r/"
# im = wrt.get_one_image(pal,par,98)
# print(im)
# imgl = Image.open(pal + im[0])
# imgr = Image.open(par + im[1])
# plt.imshow(imgr)
# plt.imshow(imgl)
# plt.show()
#按照顺序显示读取图片--------------
#-------------利用字典来，重新排序乱序的图片序列
# file = os.listdir(path)
# list = {}
# for i in file:
#     #print(i)
#     num = int(i[6:-4])
#     list[i] = num
#list= sorted(list.keys())
#list=sorted(list.items(),key=operator.itemgetter(1))
#list=sorted(list.items(),key= lambda x:x[1], reverse=False)
#print(list)
# for i in list:
#     print(i[0])
#-------------------------------
# threshold = random.sample(range(0, 400), 400)
# thre_ = []
# for i in threshold:
#     thre_.append(float(i/400))
# print(len(thre_))
#------选取字典中最大的值---------
# dic = {'a':1,'b':0,'c':6,'e':9,'l':3}
# #ed=max(dic ,key=dic.get)
# print(max(dic ,key=dic.get))
# print(dic[max(dic ,key=dic.get)])
# a = [1,9,3,5]
# b = [0.5,12,1.9,0.2]
# c = max(a)
# print(c)
# print(b[a.index((max(a)))])
# ---------------tensor_convert_to_np------------
# def convert_to_tensor(a):
#     with tf.Session() as sess:
#         b = tf.convert_to_tensor(a)
#     print(b)
#     return b
# def convert_to_arrary(a):
#     with tf.Session() as sess:
#         b = a.eval()
#     print(b)
#     return b
# a =np.array([1])
# b = convert_to_tensor(a)
# print(convert_to_arrary(b))