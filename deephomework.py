#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 19:22:24 2018

@author: jingxingjiang
"""

"""
Autoencoder编码
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_input=784
n_hidden_1=392   #（28*28）/2=392第一次autoencode层神经元个数
n_hidden_2=196   #392/2=196第二次autoencode层神经元个数
n_hidden_3=98    #第三次autoencode层神经元个数
learning_rate=0.01
batch_size=100
training_epochs = 15
display_step=1
examples_to_show=100

weights = { 
 'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name='encoder_h1'), 
 'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input]),name='decoder_h1'), 
 'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name='encoder_h2'), 
 'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]),name='decoder_h2'),
 'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]),name='encoder_h3'), 
 'decoder_h3': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2]),name='decoder_h3'),  
} 
biases = { 
 'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1]),name='encoder_b1'), 
 'decoder_b1': tf.Variable(tf.random_normal([n_input]),name='decoder_b1'), 
 'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2]),name='encoder_b2'), 
 'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1]),name='decoder_b2'),
 'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3]),name='encoder_b3'), 
 'decoder_b3': tf.Variable(tf.random_normal([n_hidden_2]),name='decoder_b3'),  
} 

X1 = tf.placeholder("float", [None, n_input]) 
X2 = tf.placeholder("float", [None, n_hidden_1]) 
X3 = tf.placeholder("float", [None, n_hidden_2]) 
y_ = tf.placeholder("float", [None,10])

def encoder1(X):
    encoder=tf.nn.sigmoid(tf.add(tf.matmul(X,weights['encoder_h1']),biases['encoder_b1']))
    return encoder

def decoder1(X):
    decoder=tf.nn.sigmoid(tf.add(tf.matmul(X,weights['decoder_h1']),biases['decoder_b1']))
    return decoder

def encoder2(X):
    encoder=tf.nn.sigmoid(tf.add(tf.matmul(X,weights['encoder_h2']),biases['encoder_b2']))
    return encoder

def decoder2(X):
    decoder=tf.nn.sigmoid(tf.add(tf.matmul(X,weights['decoder_h2']),biases['decoder_b2']))
    return decoder

def encoder3(X):
    encoder=tf.nn.sigmoid(tf.add(tf.matmul(X,weights['encoder_h3']),biases['encoder_b3']))
    return encoder

def decoder3(X):
    decoder=tf.nn.sigmoid(tf.add(tf.matmul(X,weights['decoder_h3']),biases['decoder_b3']))
    return decoder

encoder_op1=encoder1(X1)
decoder_op1=decoder1(encoder_op1)
pre1=decoder_op1
true1=X1

encoder_op2=encoder2(X2)
decoder_op2=decoder2(encoder_op2)
pre2=decoder_op2
true2=X2

encoder_op3=encoder3(X3)
decoder_op3=decoder3(encoder_op3)
pre3=decoder_op3
true3=X3

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#生成正态分布张量，第一个指标是维度，第二个指标是标准差
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

W_fc1 = weight_variable([n_hidden_3,10])
b_fc1 = bias_variable([10])
encoder_op4_last1=encoder1(X1)
encoder_op4_last12=encoder2(encoder_op4_last1)
encoder_op4_last123=encoder3(encoder_op4_last12)
y_conv=tf.nn.softmax(tf.matmul(encoder_op4_last123, W_fc1) + b_fc1)

cost1=tf.reduce_mean(tf.pow(pre1 - true1, 2))
#optimizer1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost1)
optimizer1 = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost1)
#optimizer1 = tf.train.AdagradOptimizer(learning_rate).minimize(cost1)


cost2=tf.reduce_mean(tf.pow(pre2 - true2, 2)) 
#optimizer2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost2)
optimizer2 = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost2)
#optimizer2 = tf.train.AdagradOptimizer(learning_rate).minimize(cost2)

cost3=tf.reduce_mean(tf.pow(pre3 - true3, 2)) 
optimizer3 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost3)
#optimizer3 = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost3)
#optimizer3 = tf.train.AdagradOptimizer(learning_rate).minimize(cost3)

cost4=-tf.reduce_sum(y_*tf.log(y_conv))#-tf.reduce_sum(y_*tf.log(y)),tf.reduce_mean(tf.pow(pre4 - y_, 2))
optimizer4 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost4)
#optimizer4 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost4)
#optimizer4 = tf.train.AdagradOptimizer(learning_rate).minimize(cost4)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))#给出bool值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1: 
        init = tf.initialize_all_variables() 
    else: 
        init = tf.global_variables_initializer() 
    sess.run(init) 
    
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练 
    total_batch = int(mnist.train.num_examples/batch_size) #总批数 
    #total_batch=1000
    for epoch in range(training_epochs): 
        for i in range(total_batch): 
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # max(x) = 1, min(x) = 0 
            # Run optimization op (backprop) and cost op (to get loss value) 
            c = sess.run([optimizer1, cost1], feed_dict={X1: batch_xs}) 
        if epoch % display_step == 0: 
            print("Epoch1:", '%04d' % (epoch+1), "cost=","%04f" %(c[1])) 
    print("Optimization1 Finished!") 

    for epoch in range(training_epochs): 
        for i in range(total_batch): 
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # max(x) = 1, min(x) = 0 
            # Run optimization op (backprop) and cost op (to get loss value) 
            encoder_op11=sess.run(encoder_op1, feed_dict={X1: batch_xs})
            c = sess.run([optimizer2, cost2], feed_dict={X2: encoder_op11}) 
        if epoch % display_step == 0: 
            print("Epoch2:", '%04d' % (epoch+1), "cost=","%04f" %(c[1])) 
    print("Optimization2 Finished!") 
    
    for epoch in range(training_epochs): 
        for i in range(total_batch): 
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # max(x) = 1, min(x) = 0 
            # Run optimization op (backprop) and cost op (to get loss value) 
            encoder_op11=sess.run(encoder_op1, feed_dict={X1: batch_xs})
            encoder_op22=sess.run(encoder_op2, feed_dict={X2: encoder_op11})
            c = sess.run([optimizer3, cost3], feed_dict={X3: encoder_op22}) 
        if epoch % display_step == 0: 
            print("Epoch3:", '%04d' % (epoch+1), "cost=","%04f" %(c[1])) 
    print("Optimization3 Finished!") 
    
    cost=[]
    for epoch in range(training_epochs): 
        for i in range(total_batch): 
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # max(x) = 1, min(x) = 0 
            # Run optimization op (backprop) and cost op (to get loss value) 
            c = sess.run([optimizer4, cost4], feed_dict={X1: batch_xs, y_:batch_ys}) 
        if epoch % display_step == 0: 
            print("Epoch4:", '%04d' % (epoch+1), "cost=","%04f" %(c[1])) 
            cost.append(c[1])
    print("Optimization4 Finished!")
    
    encode_decode = sess.run(accuracy, feed_dict={X1: mnist.test.images[:examples_to_show] ,y_: mnist.test.labels[:examples_to_show]})
    print ("test accuracy %g"% (encode_decode ))
    
    sess.close()

import numpy as np  
import matplotlib.pyplot as plt  
  
x=[i+1 for i in range(15)]  
plt.figure()  
plt.plot(x,cost)  
plt.xlabel("times")  
plt.ylabel("cost")  
plt.title("Training Cost Curve")  

"""
对比
"""

"""softmax回归"""
"""构建soft函数"""
x1 = tf.placeholder("float", [None, 784])#定义每一张图片的输入，占位符，不指定初值，在sess.run中设置feed_dict调用
W1 = tf.Variable(tf.zeros([784,10]))#定义权重矩阵
b1 = tf.Variable(tf.zeros([10]))#定义偏置量
hid1=tf.matmul(x1,W1)+b1
y = tf.nn.softmax(hid1)

"""损失交叉熵"""
y_ = tf.placeholder("float", [None,10])#定义每一张图片类的输入，占位符，不指定初值，在sess.run中设置feed_dict调用
cross_entropy = -tf.reduce_sum(y_*tf.log(y))#计算交叉熵
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(cross_entropy)
#梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵

"""训练测试集"""
init = tf.global_variables_initializer()#初始化我们创建的变量
sess = tf.Session()#在一个Session里面启动我们的模型，并且初始化变量
sess.run(init)
total_batch = int(mnist.train.num_examples/batch_size)
cost=[]
for epoch in range(training_epochs): 
    for i in range(total_batch): 
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c=sess.run([train_step, cross_entropy], feed_dict={x1: batch_xs, y_: batch_ys})
    if epoch % display_step == 0: 
        print("Epoch:", '%04d' % (epoch+1), "cost=","%04f" %(c[1])) 
        cost.append(c[1])
#该循环的每个步骤中，都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行

import numpy as np  
import matplotlib.pyplot as plt  
  
x=[i+1 for i in range(15)]  
plt.figure()  
plt.plot(x,cost)  
plt.xlabel("times")  
plt.ylabel("cost")  
plt.title("Training Cost Curve")  

"""模型评估"""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))#给出bool值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print ("test accuracy %g"%sess.run(accuracy, feed_dict={x1: mnist.test.images, y_: mnist.test.labels}))