#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Tue Jun  5 14:51:18 2018



@author: jingxingjiang

"""

import collections
import re  
import numpy as np
import tensorflow as tf
import functools

def getWords(file):
    lineList = []
    with open(file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        try:
            label1=float(line)
        except:
            words_=re.sub("[^a-zA-Z]+", " ",line)
            lineList.append(words_.split())
    return lineList

def getLabels(file):
    labelList=[]
    with open(file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        try:
            label1=float(line)
            if label1>=0 and label1<=0.2:
                labelList.append([1,0,0,0,0])
            if label1>0.2 and label1<=0.4:
                labelList.append([0,1,0,0,0])
            if label1>0.4 and label1<=0.6:
                labelList.append([0,0,1,0,0])
            if label1>0.6 and label1<=0.8:
                labelList.append([0,0,0,1,0])
            if label1>0.8 and label1<=1.0:
                labelList.append([0,0,0,0,1])
        except:
            continue
    labelList = np.array(labelList)
    return labelList

def glove_word2vec(file):
    word2vecList={}
    with open(file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line1=line.rstrip().split()
        word2vecList[line1[0]]=[float(i) for i in line1[1:]]
    return word2vecList

def words2Array(lineList,word2vecList):
    linesArray=[]
    wordsArray=[]
    steps = []
    for line in lineList:
        t = 0
        p = 0
        for i in range(MAX_SIZE):
            if i<len(line):
                try:
                    wordsArray.append(word2vecList[line[i]])
                    p = p + 1
                except KeyError:
                    t=t+1
                    continue
            else:
               wordsArray.append(np.array([0.0]*dimsh))
        for i in range(t):
            wordsArray.append(np.array([0.0]*dimsh))
        steps.append(p)
        linesArray.append(wordsArray)
        wordsArray = []
    linesArray = np.array(linesArray)
    steps = np.array(steps)
    return linesArray, steps

MAX_SIZE=25
dimsh=100
num_nodes = 128
batch_size = 100
output_size = 5

trainwords=getWords(r'D:\stanfordSentimentTreebank\train1.txt')
trainlabels=getLabels(r'D:\stanfordSentimentTreebank\train1.txt')

validwords=getWords(r'D:\stanfordSentimentTreebank\valid1.txt')
validlabels=getLabels(r'D:\stanfordSentimentTreebank\valid1.txt')

testwords=getWords(r'D:\stanfordSentimentTreebank\test1.txt')
testlabels=getLabels(r'D:\stanfordSentimentTreebank\test1.txt')

word2vecList=glove_word2vec(r'D:\glove100d.txt')
trainvec,trainSteps=words2Array(trainwords,word2vecList)
testvec,testSteps=words2Array(testwords,word2vecList)

from tensorflow.contrib import rnn 

#单向LSTM
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,MAX_SIZE,dimsh))
    tf_train_steps = tf.placeholder(tf.int32,shape=(batch_size))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,output_size))

    tf_test_dataset = tf.constant(testvec,tf.float32)
    tf_test_steps = tf.constant(testSteps,tf.int32)

    lstm_cell = rnn.BasicLSTMCell(num_units = num_nodes,state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.7)
    #将输出为128维的结果压缩为5维的对比输出结果
    w1 = tf.Variable(tf.truncated_normal([num_nodes,num_nodes // output_size], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([num_nodes // output_size], stddev=0.1))

    w2 = tf.Variable(tf.truncated_normal([num_nodes // output_size, output_size], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([output_size], stddev=0.1))

    def model(dataset, steps):
        outputs, last_states = tf.nn.dynamic_rnn(cell = lstm_cell,
                                                 dtype = tf.float32,
                                                 sequence_length = steps,
                                                 inputs = dataset)

        hidden = last_states[-1]#last_states代表最后一个神经元输出，包含：细胞状态c与输出h
        hidden = tf.matmul(hidden, w1) + b1
        logits = tf.matmul(hidden, w2) + b2
        return logits

    train_logits = model(tf_train_dataset, tf_train_steps)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=train_logits))
    optimizer = tf.train.GradientDescentOptimizer(0.03).minimize(loss)
    test_prediction = tf.nn.softmax(model(tf_test_dataset, tf_test_steps))

#双向LSTM
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,MAX_SIZE,dimsh))
    tf_train_steps = tf.placeholder(tf.int32,shape=(batch_size))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,output_size))

    tf_test_dataset = tf.constant(testvec,tf.float32)
    tf_test_steps = tf.constant(testSteps,tf.int32)

    cell_fw = rnn.BasicLSTMCell(num_units = num_nodes,state_is_tuple=True)
    cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=0.6) 
    cell_bw = rnn.BasicLSTMCell(num_units = num_nodes,state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=0.6)

    def model(cell_fw,cell_bw,dataset, steps):
        outputs, last_states = tf.nn.bidirectional_dynamic_rnn(  
                                                 cell_fw=cell_fw,  
                                                 cell_bw=cell_bw,  
                                                 inputs=dataset,  
                                                 sequence_length=steps,  
                                                 dtype=tf.float32) 
        outputs_fw, outputs_bw = outputs
        outputs = tf.concat([outputs_fw, outputs_bw], axis=2)  
        outputs = tf.reduce_mean(outputs, axis=1)  
        outputs = tf.contrib.layers.fully_connected(  
                    inputs=outputs,  
                    num_outputs=output_size,  
                    activation_fn=None)  
        return outputs
    train_logits = model(cell_fw,cell_bw,tf_train_dataset, tf_train_steps)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=train_logits))
    optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
    test_prediction = tf.nn.softmax(model(cell_fw,cell_bw,tf_test_dataset, tf_test_steps))

num_steps = 6100
summary_frequency = 500

with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        offset = (step * batch_size) % (len(trainlabels)-batch_size)
        feed_dict={tf_train_dataset:trainvec[offset:offset + batch_size],
                   tf_train_labels:trainlabels[offset:offset + batch_size],
                   tf_train_steps:trainSteps[offset:offset + batch_size]}
        _, l = session.run([optimizer,loss],feed_dict = feed_dict)
        mean_loss += l
        if step >0 and step % summary_frequency == 0:
            mean_loss = mean_loss / summary_frequency
            print("The step is: %d"%(step))
            print("In train data,the loss is:%.4f"%(mean_loss))
            mean_loss = 0
            acrc = 0
            prediction = session.run(test_prediction)
            for i in range(len(prediction)):
                if [k for k in prediction[i]].index(max(prediction[i]))==[j for j in testlabels[i]].index(1):
                    acrc = acrc + 1
            print("In test data,the accuracy is:%.2f%%"%((acrc/len(testlabels))*100))
