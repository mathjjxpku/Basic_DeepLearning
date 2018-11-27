#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:51:38 2018

@author: jingxingjiang
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np  
import os
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages
size1=224
root = '/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework2/'

TFwriter = tf.python_io.TFRecordWriter("/Users/jingxingjiang/大数据班课程资料/深度学习课程/data/faceTF.tfrecords")

for className in os.listdir(root)[1:]:
    label = int(className[0:])
    classPath = root+"/"+className+"/"
    for parent, dirnames, filenames in os.walk(classPath):
        for filename in filenames:
            imgPath = classPath+"/"+filename
            print (imgPath)
            img = Image.open(imgPath)
            img= img.resize((size1,size1))
            print (img.size,img.mode)
            imgRaw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label":tf.train.Feature(int64_list = tf.train.Int64List(value=[label])),
                "img":tf.train.Feature(bytes_list = tf.train.BytesList(value=[imgRaw]))
            }) )
            TFwriter.write(example.SerializeToString())

TFwriter.close()

fileNameQue = tf.train.string_input_producer(["/Users/jingxingjiang/大数据班课程资料/深度学习课程/data/faceTF.tfrecords"])
reader = tf.TFRecordReader()
key,value = reader.read(fileNameQue)
features = tf.parse_single_example(value,features={ 'label': tf.FixedLenFeature([], tf.int64),
                                           'img' : tf.FixedLenFeature([], tf.string),})

img = tf.decode_raw(features["img"], tf.uint8)
label = tf.cast(features["label"], tf.int32)

def get_batch(img,label,batch_size):
    img=tf.reshape(img,[size1,size1,3])
    img_batch,label_batch = tf.train.shuffle_batch([img,label],batch_size=batch_size, capacity=1500,min_after_dequeue=800)
    
    img_batch=tf.subtract(tf.cast(img_batch,tf.float32),128)
    img_batch=tf.divide(img_batch,255)#归一化，一个像素点的值最多255
    
    label_batch=tf.one_hot(label_batch,17,1,0)
    label_batch= tf.cast(label_batch, tf.float64)
    return img_batch,label_batch

"""卷积神经网络"""
trainx_x,train_y=get_batch(img,label,128)
"""权重初始化"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)#生成正态分布张量，第一个指标是维度，第二个指标是标准差
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

"""卷积与池化"""
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=(None,size1,size1,3))
y_ = tf.placeholder("float", shape=[None, 17])
is_training = tf.placeholder(tf.bool)

"""第一层卷积"""
W_conv11 = weight_variable([3, 3, 3, 64])
b_conv11 = bias_variable([64])

x = tf.layers.batch_normalization(x, training=is_training)

h_conv11 = tf.nn.relu(conv2d(x, W_conv11) + b_conv11)

W_conv12 = weight_variable([3, 3, 64, 64])
b_conv12 = bias_variable([64])

h_conv11 = tf.layers.batch_normalization(h_conv11, training=is_training)

h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12)
h_pool1 = max_pool_2x2(h_conv12)

"""第二层卷积"""
W_conv21 = weight_variable([3, 3, 64, 128])
b_conv21 = bias_variable([128])

h_pool1 = tf.layers.batch_normalization(h_pool1, training=is_training)

h_conv21 = tf.nn.relu(conv2d(h_pool1, W_conv21) + b_conv21)

W_conv22 = weight_variable([3, 3, 128, 128])
b_conv22 = bias_variable([128])

h_conv21 = tf.layers.batch_normalization(h_conv21, training=is_training)

h_conv22 = tf.nn.relu(conv2d(h_conv21, W_conv22) + b_conv22)
h_pool2 = max_pool_2x2(h_conv22)

"""第三层卷积"""
W_conv31 = weight_variable([3, 3, 128, 256])
b_conv31 = bias_variable([256])

h_pool2 = tf.layers.batch_normalization(h_pool2, training=is_training)

h_conv31 = tf.nn.relu(conv2d(h_pool2, W_conv31) + b_conv31)

W_conv32 = weight_variable([3, 3, 256, 256])
b_conv32 = bias_variable([256])

h_conv31 = tf.layers.batch_normalization(h_conv31, training=is_training)

h_conv32 = tf.nn.relu(conv2d(h_conv31, W_conv32) + b_conv32)

W_conv33 = weight_variable([3, 3, 256, 256])
b_conv33 = bias_variable([256])

h_conv32 = tf.layers.batch_normalization(h_conv32, training=is_training)

h_conv33 = tf.nn.relu(conv2d(h_conv32, W_conv33) + b_conv33)

h_pool3 = max_pool_2x2(h_conv33)

"""第四层卷积"""
W_conv41 = weight_variable([3, 3, 256, 512])
b_conv41 = bias_variable([512])

h_pool3 = tf.layers.batch_normalization(h_pool3, training=is_training)

h_conv41 = tf.nn.relu(conv2d(h_pool3, W_conv41) + b_conv41)

W_conv42 = weight_variable([3, 3, 512, 512])
b_conv42 = bias_variable([512])

h_conv41 = tf.layers.batch_normalization(h_conv41, training=is_training)

h_conv42 = tf.nn.relu(conv2d(h_conv41, W_conv42) + b_conv42)

W_conv43 = weight_variable([3, 3, 512, 512])
b_conv43 = bias_variable([512])

h_conv42 = tf.layers.batch_normalization(h_conv42, training=is_training)

h_conv43 = tf.nn.relu(conv2d(h_conv42, W_conv43) + b_conv43)
h_pool4 = max_pool_2x2(h_conv43)

"""第五层卷积"""
W_conv51 = weight_variable([3, 3, 512, 512])
b_conv51 = bias_variable([512])

h_pool4 = tf.layers.batch_normalization(h_pool4, training=is_training)

h_conv51 = tf.nn.relu(conv2d(h_pool4, W_conv51) + b_conv51)

W_conv52 = weight_variable([3, 3, 512, 512])
b_conv52 = bias_variable([512])

h_conv51 = tf.layers.batch_normalization(h_conv51, training=is_training)

h_conv52 = tf.nn.relu(conv2d(h_conv51, W_conv52) + b_conv52)

W_conv53 = weight_variable([3, 3, 512, 512])
b_conv53 = bias_variable([512])

h_conv52 = tf.layers.batch_normalization(h_conv52, training=is_training)

h_conv53 = tf.nn.relu(conv2d(h_conv52, W_conv53) + b_conv53)
h_pool5 = max_pool_2x2(h_conv53)

"""密集连接层"""
W_fc1 = weight_variable([7 * 7 * 512, 4096])
b_fc1 = bias_variable([4096])

h_pool5_flat = tf.reshape(h_pool5, [-1, 7 * 7 * 512])#-1代表由实际情况决定
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

W_fc12 = weight_variable([4096, 4096])
b_fc12 = bias_variable([4096])
h_fc12 = tf.nn.relu(tf.matmul(h_fc1, W_fc12) + b_fc12)

"""Dropout减少过拟合"""
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc12, keep_prob)

"""输出层"""
W_fc2 = weight_variable([4096, 17])
b_fc2 = bias_variable([17])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

"""卷积神经网络效果评估"""
cross_entropy = -tf.reduce_mean(y_*tf.log(y_conv))#损失函数
train_step = tf.train.AdamOptimizer(1e-1).minimize(cross_entropy)#损失函数最小化
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))#训练测试对比
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#正确率均值

init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    
    for i in range(8):
        batch_x,batch_y=sess.run([trainx_x, train_y])
        c1=sess.run(cross_entropy, feed_dict={x: batch_x, y_:batch_y,keep_prob: 0.8, is_training:True}) 
        print(c1)
    
    batch_x1,batch_y1=sess.run([trainx_x, train_y])
    encode_decode = sess.run(accuracy, feed_dict={x: batch_x1 ,y_: batch_y1,keep_prob: 0.8 ,is_training:False})
    print ("test accuracy %g"% (encode_decode ))
    
    coord.request_stop()
    coord.join(threads)