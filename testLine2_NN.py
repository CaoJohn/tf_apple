#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:59:34 2017

@author: john
"""

import tensorflow as tf
import numpy as np
import csv
import time

#from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#input_ = {"raidType":None,"rnd_io_cnt_ratio":None,"seq_io_cnt_ratio":None,"rw_ratio":None}
#output1_ = {"rnd_rt":None,"seq_rt":None}




input_ = []
output1_ = []
output2_ = []
data_lenth = 1000
with open('train_5.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        num_row = [ float(i) for i in row ]
        input_.append(num_row[0:9])
        output1_.append(num_row[9:10])
        output2_.append(num_row[10:11])
#print(input_)
#print(output1_)


#trX = np.linspace(-1, 1, 101)
#trY1 = np.arange(4).reshape(2,2)
my_X = np.array(input_[0:data_lenth])
my_Y1 = np.array(output1_[0:data_lenth])
my_Y2 = np.array(output2_[0:data_lenth])

scaler_x = StandardScaler().fit(my_X)
scaler_y1 = StandardScaler().fit(my_Y1)
scaler_y2 = StandardScaler().fit(my_Y2)

trX = scaler_x.transform(my_X)
trY1 = scaler_y1.transform(my_Y1)
trY2 = scaler_y1.transform(my_Y2)


print("********* traning raw input data ******")
print(my_X)

print("********* traning raw response_time_rnd ******")
print(my_Y1)

print("********* traning raw response_time_seq ******")
print(my_Y2)
print("********* starting  normalize ******")
time.sleep(2)
print("*********  normalize  input data ******")
print(trX)
print("********* normalize response_time_rnd ******")
print(trY1)
print("********* normalize response_time_seq ******")
print(trY2)


#trY1 = trW * trX + np.random.rand(*trX.shape) * 0.123 
# 创建两个占位符，数据类型是 tf.float32
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
biases = tf.Variable(tf.zeros(1) + 0.1)
# 创建一个变量系数 w , 最后训练出来的值，应该接近 2 
w = tf.Variable(tf.zeros([1, 9]), name = "weights")
y_model = tf.multiply(X, w)+biases
# 定义损失函数 (Y - y_model)^2
cost = tf.square(Y - y_model)
# 定义学习率
learning_rate = 0.01
# 使用梯度下降来训练模型，学习率为 learning_rate , 训练目标是使损失函数最小
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:  
  # 初始化所有的变量 
  init = tf.global_variables_initializer()
  sess.run(init) 
  # 对模型训练100次 
  for i in xrange(100): 
    for (x, y) in zip(trX, trY1): 
      sess.run(train_op, feed_dict = {X: x, Y: y}) 
  # 输出 w 的值 
  W = sess.run(w)

  # 输出 b 的值 
  B = sess.run(biases)

#test_input = np.transpose(np.transpose(np.array([100,5000,4000])))
test_input = np.transpose(np.transpose(np.array([50,100,5000,4000,70,20,40,90,1000000])))
W = np.transpose(W)

print("************* testing response_time_rnd ****************")
print(W)
print(B)
print("testing data: ")
print(test_input)
test_X = scaler_x.transform(test_input)
print(test_X)
test_output = np.dot(test_X,W)+B
print("result data(response_time_rnd): ")
print(scaler_y1.inverse_transform(test_output))



with tf.Session() as sess:  
  # 初始化所有的变量 
  init = tf.global_variables_initializer()
  sess.run(init) 
  # 对模型训练100次 
  for i in xrange(100): 
    for (x, y) in zip(trX, trY2): 
      sess.run(train_op, feed_dict = {X: x, Y: y}) 
  # 输出 w 的值 
  W = sess.run(w)
  # 输出 b 的值 
  B = sess.run(biases)
  
W = np.transpose(W)
print("************* testing  response_time_seq ****************")
print(W)
print(B)
print("testing data: ")
print(test_input)
test_X = scaler_x.transform(test_input)
print(test_X)
test_output = np.dot(test_X,W)+B
print("result data(response_time_seq): ")
print(scaler_y1.inverse_transform(test_output))


