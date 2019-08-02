import tensorflow as tf
import cv2
import os
import random
import operator
from PIL import Image
import numpy as np


def write_tfrecords(image_flie,label_file,save_dir,name):
    filename = (save_dir + name + '.tfrecords')
    write = tf.python_io.TFRecordWriter(filename)
    dictionary = {}
    print('\nTransform start......')
    for i in os.listdir(image_flie):  # listdir的参数是文件夹的路径
        value_num = int(i[6:-4])
        dictionary[i] = value_num
    list = sorted(dictionary.items(), key=operator.itemgetter(1))
    threshold = random.sample(range(0, len(list)),len(list))
    for fl in threshold:
        fl = int(fl)
        im_name = list[fl]
        #print(im_name)
        fil = image_flie + im_name[0]
        print(fil)
        image = cv2.imread(fil)
        image = cv2.resize(image, (225, 225))
        b, g, r = cv2.split(image)
        rgb_image = cv2.merge([r, g, b])
        image_raw = rgb_image.tostring()
        list_ = []
        with open(label_file, 'r') as f:
            for i in f:
                i = i.strip()
                i = i.replace('\n', ' ')
                i = i.split(" ")
                # print(i)
                list_.append(i)
            label = int(list_[fl][0])  # label一个数字
            print(label)
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
            }))
        write.write(example.SerializeToString())
    write.close()
    #生成对应的tfrecords文件
    print(len(list))
    print('Transform done!')

def read_records(file, b_size):
    #file为相应的tfrecords文件
    filename_queue = tf.train.string_input_producer([file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.float32),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [225, 225, 3])
    label = tf.cast(img_features['label'], tf.float32)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=b_size, num_threads=1, capacity=20)
    return image_batch, label_batch
#在0-1上产生400个不同的点
def produce_threshold():
    threshold = random.sample(range(0, 50), 50)
    thre_ = []
    for i in threshold:
        thre_.append(float(i/50))
    return thre_
#img_l和img_r为对应的图片路径，num则是要取出的图片编号
def get_one_image(img_l,img_r,num):
    image_l = []
    image_r = []
    for i in os.listdir(img_l):
        if i[6:-4] == str(num):
            image_l.append(img_l+i)
    for y in os.listdir(img_r):
        if y[6:-4] == str(num):
            image_r.append(img_r+y)
    return image_l,image_r
#单张图片预处理
def prepare(pic_dir):
    image = Image.open(pic_dir)
    image_a = image.resize([225,225])
    image_b = np.array(image_a)
    image_c = tf.cast(image_b,tf.float32)
    image_d = tf.image.per_image_standardization(image_c)
    image_e = tf.reshape(image_d,[1,225,225,3])
    return image_e

def to_tensor(img):
    image_c = tf.cast(img, tf.float32)
    image_d = tf.image.per_image_standardization(image_c)
    image_e = tf.reshape(image_d, [1, 225, 225, 3])
    return image_e

def get_pic(image_flie_l,image_flie_r,num):
    dictionary_l = {}
    for i in os.listdir(image_flie_l):  # listdir的参数是文件夹的路径
        value_num = int(i[6:-4])
        dictionary_l[i] = value_num
    list_l = sorted(dictionary_l.items(), key=operator.itemgetter(1))
    fil_l = image_flie_l + '/'+ list_l[num][0]
    image_l = cv2.imread(fil_l)
    # print(type(image_l))<class 'numpy.ndarray'>
    img_l = cv2.resize(image_l, (225, 225))
    # img_l= to_tensor(image_l)
    
    dictionary_r = {}
    for i in os.listdir(image_flie_r):  # listdir的参数是文件夹的路径
        value_num = int(i[6:-4])
        dictionary_r[i] = value_num
    list_r = sorted(dictionary_r.items(), key=operator.itemgetter(1))
    fil_r = image_flie_r + '/'+ list_r[num][0]
    # print(fil_r)
    image_r = cv2.imread(fil_r)
    # print(type(image_r))
    img_r = cv2.resize(image_r, (225, 225))
    # img_r = to_tensor(image_r)
    return img_l,img_r

# def nparray(image_flie_l,image_flie_r,num):
