# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:40:37 2018

@author: v-jinji
"""

def delblankline(infile1,infile2,trainfile,validfile,testfile):
 #### 2 是test，3 是valid的我写错了
    info1 = open(infile1,'r')
    info2 = open(infile2,'r')
    train=open(trainfile,'w')
    valid=open(validfile,'w')
    test=open(testfile,'w')
    lines1 = info1.readlines()
    lines2 = info2.readlines()
    for i in range(1,len(lines1)):
        t1=lines1[i].replace("-LRB-","(")
        t2=t1.replace("-RRB-",")")
        ###把括号部分还原
        k=lines2[i].strip().split(",")
        t=t2.strip().split('\t')
        if k[1]=='1':
            train.writelines(t[1])
            train.writelines("\n")
        elif(k[1]=='3'):
            valid.writelines(t[1])
            valid.writelines("\n")
        elif(k[1]=='2'):
            test.writelines(t[1])
            test.writelines("\n")
    print("end")
    info1.close()
    info2.close()
    train.close()
    valid.close()
    test.close()       


def tag(infile1,infile2,outputfile3):
    info1 = open(infile1,'r')
    info2 = open(infile2,'r')
    info3=open(outputfile3,'w')
    lines1 = info1.readlines()
    lines2 = info2.readlines()
    text={}
    for i in range(0,len(lines1)):
        s=lines1[i].strip().split("|")
        text[s[1]]=s[0]
    for j in range(1,len(lines2)):
        k=lines2[j].strip().split("|")
        if(k[0] in text):
            info3.writelines(text[k[0]])
            info3.writelines("\n")
            info3.writelines(k[1])
            info3.writelines("\n")
    print("end2d1")
    info1.close()
    info2.close()
    info3.close()
             
def tag1(infile0,infile1,infile2,infile3,infile4,infile5,infile6):
    info0 = open(infile0,'r')
    info1 = open(infile1,'r')
    info2 = open(infile2,'r')
    info3 = open(infile3,'r')
    info4 = open(infile4,'w')
    info5 = open(infile5,'w')
    info6 = open(infile6,'w')
    lines0 = info0.readlines()
    lines1 = info1.readlines()
    lines2 = info2.readlines()
    lines3 = info3.readlines()   
    for i in range(0,len(lines0),2):
        if  lines0[i] in lines1:
            info4.writelines(lines0[i])
            info4.writelines(lines0[i+1])
        if  lines0[i] in lines2:
            info5.writelines(lines0[i])
            info5.writelines(lines0[i+1])
        if lines0[i] in lines3:
            info6.writelines(lines0[i])
            info6.writelines(lines0[i+1])
  
    print("end3d1")
    info0.close()
    info1.close()
    info2.close()
    info3.close()
    info4.close()
    info5.close()
    info6.close()

delblankline('/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/stanfordSentimentTreebank/datasetSentences.txt',"/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/stanfordSentimentTreebank/datasetSplit.txt","/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/data/train.txt","/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/data/valid.txt","/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/data/test.txt")
tag("/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/stanfordSentimentTreebank/dictionary.txt","/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/stanfordSentimentTreebank/sentiment_labels.txt","/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/data/allsentimet.txt")
tag1("/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/data/allsentimet.txt","/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/data/train.txt","/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/data/valid.txt","/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/data/test.txt","/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/data/train1.txt","/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/data/valid1.txt","/Users/jingxingjiang/大数据班课程资料/深度学习课程/homework/homework3/data/test1.txt")
