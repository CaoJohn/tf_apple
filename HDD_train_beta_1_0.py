#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:22:30 2017

@author: john
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import csv
import time
from sklearn.preprocessing import StandardScaler
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QLabel

class Ui_Form(QWidget):
    def setupUi(self, Form):
        Form.setObjectName("HDD training")
        Form.resize(800, 600)
        self.combo_list = ["read_size","write_size"]
        self.lenth = 1000
#        self.pushButton = QtWidgets.QPushButton(Form)
#        self.pushButton.setGeometry(QtCore.QRect(70, 50, 75, 23))
#        self.pushButton.setObjectName("pushButton")
        
        self.pushButton_input = QtWidgets.QPushButton(Form)
        self.pushButton_input.setGeometry(QtCore.QRect(80, 100, 200, 80))
        self.pushButton_input.setObjectName("input")
        
        self.pushButton_train = QtWidgets.QPushButton(Form)
        self.pushButton_train.setGeometry(QtCore.QRect(80, 200, 200, 80))
        self.pushButton_train.setObjectName("train")
        
        self.pushButton_run = QtWidgets.QPushButton(Form)
        self.pushButton_run.setGeometry(QtCore.QRect(400, 150, 60, 40))
        self.pushButton_run.setObjectName("result")
        
        self.pushButton_draw = QtWidgets.QPushButton(Form)
        self.pushButton_draw.setGeometry(QtCore.QRect(450, 500, 100, 40))
        self.pushButton_draw.setObjectName("draw")
        
        self.pushButton_draw_3d = QtWidgets.QPushButton(Form)
        self.pushButton_draw_3d.setGeometry(QtCore.QRect(560, 500, 100, 40))
        self.pushButton_draw_3d.setObjectName("draw3D")
        
        self.pushButton_draw_compare = QtWidgets.QPushButton(Form)
        self.pushButton_draw_compare.setGeometry(QtCore.QRect(670, 500, 100, 40))
        self.pushButton_draw_compare.setObjectName("compare")
        
        # Create textbox
        self.textbox = QLineEdit(Form)
        self.textbox.setGeometry(QtCore.QRect(400, 100, 300, 40))
        self.textbox.setObjectName("result:")
        
        self.textbox_one = QLineEdit(Form)
        self.textbox_one.setGeometry(QtCore.QRect(120, 500, 300, 40))
        self.textbox_one.setObjectName("result:")
        
        # Create label
        self.train_label = QLabel(Form)
        self.train_label.setGeometry(QtCore.QRect(120, 280, 600, 200))
        self.train_label.setObjectName("train")
        
        # Create label
        self.result_label = QLabel(Form)
        self.result_label.setGeometry(QtCore.QRect(400, 240, 600, 200))
        self.result_label.setObjectName("result")
        
        # Create combobox
        self.combo_box = QtWidgets.QComboBox(Form)
        self.combo_box.setGeometry(QtCore.QRect(50, 500, 60, 40))
        self.combo_box.setObjectName("combo")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        
        self.pushButton_input.clicked.connect(self.openFile)
        self.pushButton_train.clicked.connect(self.train)
        self.pushButton_run.clicked.connect(self.run)
        self.pushButton_draw.clicked.connect(self.drawOne)
        self.pushButton_draw_3d.clicked.connect(self.draw3D)
        self.pushButton_draw_compare.clicked.connect(self.compare)
        

        
#        self.connect( self.pushButton_input, QtCore.SIGNAL( 'clicked()' ), self.openFile )

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "HDD training"))
#        self.pushButton.setText(_translate("Form", "PushButton"))
        self.pushButton_input.setText(_translate("Form", "open"))
        self.pushButton_train.setText(_translate("Form", "train"))
        self.pushButton_run.setText(_translate("Form", "run"))
        self.pushButton_draw.setText(_translate("Form", "draw"))
        self.pushButton_draw_3d.setText(_translate("Form", "draw3D"))
        self.pushButton_draw_compare.setText(_translate("Form", "compare"))
        self.textbox.setText(_translate("Form", "80,100,5000,4000,70,80,40,90,1000000"))
        self.textbox_one.setText(_translate("Form", "3000,3100,3200,3300,3400,3500,3600,3700,3800"))
        self.combo_box.addItems(self.combo_list)
#        self.train_label.setText(_translate("Form", "train:"))
#        self.result_label.setText(_translate("Form", "result:"))
        
        
    def openFile(self):
        print("get file")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if self.fileName:
            print(self.fileName)
            
    def compare(self):
        print("compare")
        plt.close("all")
        rnd_result = []
        seq_result = []
        test_data = self.textbox_one.text()
        test_data = np.arange(len(self.raw_input))
        output_1 = self.raw_output_1
        output_2 = self.raw_output_2
        print("************* testing input ****************")
#        test_array = np.array([ float(i) for i in test_data.split(",")])
        test_array = self.raw_input
        for i in test_array:
            test_input = i
            print("************* testing response_time_rnd ****************")
            print("testing data: ")
            print(test_input)
            test_X = self.scaler_x.transform(test_input)
            test_output = np.dot(test_X,self.W1)+self.B1
            rnd_result.append(self.scaler_y1.inverse_transform(test_output))
            
            print("************* testing response_time_seq ****************")
            print("testing data: ")
            print(test_input)
            test_X = self.scaler_x.transform(test_input)
            test_output = np.dot(test_X,self.W2)+self.B2
            seq_result.append(self.scaler_y2.inverse_transform(test_output))
        print("result data(response_time_rnd): ")
        print(rnd_result)
        print("result data(response_time_seq): ")
        print(seq_result)
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(test_data, rnd_result)
        axarr[0].set_title('prediction')
        axarr[1].scatter(test_data, seq_result)
        axarr[0].set_xlabel("reads")
        axarr[1].set_xlabel("reads")
        axarr[0].set_ylabel("rnd")
        axarr[1].set_ylabel("seq")
        f, axarr_real = plt.subplots(2, sharex=True)
        axarr_real[0].plot(test_data, output_1)
        axarr_real[0].set_title('real')
        axarr_real[1].scatter(test_data, output_2)
        axarr_real[0].set_xlabel("reads")
        axarr_real[1].set_xlabel("reads")
        axarr_real[0].set_ylabel("rnd")
        axarr_real[1].set_ylabel("seq")
        plt.show()
            
    def draw3D(self):
        print("draw 3D")
        rnd_result = []
        seq_result = []
#        read_array = np.linspace(self.input_min[2]+(self.input_max[2]-self.input_min[2])/4,self.input_max[2]-(self.input_max[2]-self.input_min[2])/4,50)
#        write_array = np.linspace(self.input_min[3]+(self.input_max[3]-self.input_min[3])/4,self.input_max[3]-(self.input_max[3]-self.input_min[3])/4,50)
        input3d = self.raw_input[self.lenth/4:self.lenth*3/4:2]
        read_array = input3d[:,2]
        write_array = input3d[:,3]
        print(input3d)
        test_input = self.input_mean
        for i in range(len(input3d[:,2])):
#            test_input[2] = read_array[i]
#            test_input[3] = write_array[i]
            test_input[2] = np.transpose(input3d[:,2][i])
            test_input[3] = np.transpose(input3d[:,3][i])
            print("************* testing response_time_rnd ****************")
            print("testing data: ")
            print(test_input)
            test_X = self.scaler_x.transform(test_input)
            test_output = np.dot(test_X,self.W1)+self.B1
            rnd_result.append(self.scaler_y1.inverse_transform(test_output).tolist()[0])
            
            print("************* testing response_time_seq ****************")
            print("testing data: ")
            print(test_input)
            test_X = self.scaler_x.transform(test_input)
            test_output = np.dot(test_X,self.W2)+self.B2
            seq_result.append(self.scaler_y2.inverse_transform(test_output).tolist()[0]) 
        print("result data(response_time_rnd): ")
        print(rnd_result)
        print("result data(response_time_seq): ")
        print(seq_result)
        
        # new a figure and set it into 3d
        plt.close("all")
        fig = plt.figure()
#        ax = Axes3D(fig)
        
        # set figure information
        
        # draw the figure, the color is r = read
        #figure = ax.plot(trX1, trX2, trY,  c='r')
        # ############ first subplot ############
        print("************* check ***************")
        
        print(read_array.tolist())
        print(write_array.tolist())
        print(rnd_result)
        print(seq_result)
        print(type(read_array.tolist()))
        print(type(read_array.tolist()[0]))
        
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        ax.set_title("r/w--->rt_rnd")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
#        ax.plot_trisurf(read_array.tolist(), write_array.tolist(), rnd_result) 
        ax.scatter(read_array, write_array, np.array(rnd_result))
        # ############ second subplot ############
        ax = fig.add_subplot(2, 1, 2, projection='3d')
        ax.set_title("r/w--->rt_seq")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
#        ax.plot_trisurf(read_array, write_array,seq_result)
        ax.scatter(read_array, write_array,np.array(seq_result))
        plt.show()
        
        
    def drawOne(self):
        print("draw one")
        which_one = unicode(self.combo_box.currentText())
        print(which_one)
        rnd_result = []
        seq_result = []
        if which_one == "read_size" :
            plt.close("all")
            test_data = self.textbox_one.text()
            print("************* testing input ****************")
            test_array = np.array([ float(i) for i in test_data.split(",")])
            test_input = self.input_mean
            for i in test_array:
                test_input[2] = i
                print("************* testing response_time_rnd ****************")
                print("testing data: ")
                print(test_input)
                test_X = self.scaler_x.transform(test_input)
                test_output = np.dot(test_X,self.W1)+self.B1
                rnd_result.append(self.scaler_y1.inverse_transform(test_output))
                
                print("************* testing response_time_seq ****************")
                print("testing data: ")
                print(test_input)
                test_X = self.scaler_x.transform(test_input)
                test_output = np.dot(test_X,self.W2)+self.B2
                seq_result.append(self.scaler_y2.inverse_transform(test_output))
            print("result data(response_time_rnd): ")
            print(rnd_result)
            print("result data(response_time_seq): ")
            print(seq_result)
            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].plot(test_array, rnd_result)
            axarr[0].set_title('Sharing X axis')
            axarr[1].scatter(test_array, seq_result)
            axarr[0].set_xlabel("reads")
            axarr[1].set_xlabel("reads")
            axarr[0].set_ylabel("rnd")
            axarr[1].set_ylabel("seq")
            plt.show()
            
        elif which_one == "write_size" :
            print("write")
            plt.close("all")
            test_data = self.textbox_one.text()
            print("************* testing input ****************")
            test_array = np.array([ float(i) for i in test_data.split(",")])
            test_input = self.input_mean
            for i in test_array:
                test_input[3] = i
                print("************* testing response_time_rnd ****************")
                print("testing data: ")
                print(test_input)
                test_X = self.scaler_x.transform(test_input)
                test_output = np.dot(test_X,self.W1)+self.B1
                rnd_result.append(self.scaler_y1.inverse_transform(test_output))
                
                print("************* testing response_time_seq ****************")
                print("testing data: ")
                print(test_input)
                test_X = self.scaler_x.transform(test_input)
                test_output = np.dot(test_X,self.W2)+self.B2
                seq_result.append(self.scaler_y2.inverse_transform(test_output))
            print("result data(response_time_rnd): ")
            print(rnd_result)
            print("result data(response_time_seq): ")
            print(seq_result)
            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].plot(test_array, rnd_result)
            axarr[0].set_title('Sharing X axis')
            axarr[1].scatter(test_array, seq_result)
            axarr[0].set_xlabel("writes")
            axarr[1].set_xlabel("writes")
            axarr[0].set_ylabel("rnd")
            axarr[1].set_ylabel("seq")
            plt.show()
            
        else:
            print("combo box is wrong")
        
            
    def run(self):
        print("run")
        test_data = self.textbox.text()
        print("************* testing input ****************")
        test_array = np.array([ float(i) for i in test_data.split(",")])
        print(test_array)
        for i in range(0,len(test_array)):
            if test_array[i]>self.input_max[i] or test_array[i] < self.input_min[i]:
                print("input is abnormal!!!")
                QtWidgets.QMessageBox.information( self, "Pyqt", "input data is abnormal" )
                return 1
        print("input is OK")
        
        print("************* testing response_time_rnd ****************")
        print("W1:")
        print(self.W1)
        print("B1:")
        print(self.B1)
        print("testing data: ")
        print(test_array)
        test_X = self.scaler_x.transform(test_array)
        test_output = np.dot(test_X,self.W1)+self.B1
        rnd_output = self.scaler_y1.inverse_transform(test_output)
        print("result data(response_time_rnd): ")
        print(rnd_output)
        
        
        print("************* testing  response_time_seq ****************")
        print("W2:")
        print(self.W2)
        print("B2:")
        print(self.B2)
        print("testing data: ")
        print(test_array)
        test_X = self.scaler_x.transform(test_array)
        test_output = np.dot(test_X,self.W2)+self.B2
        seq_ouput = self.scaler_y2.inverse_transform(test_output)
        print("result data(response_time_seq): ")
        print(seq_ouput)
        
        self.train_label.setText("response_time_rnd:"+str(rnd_output)+"\nresponse_time_seq:"+str(seq_ouput))
                  
    def train(self):
        print("train")
        input_ = []
        output1_ = []
        output2_ = []
        data_lenth = self.lenth
        with open(self.fileName) as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            for row in f_csv:
                num_row = [ float(i) for i in row ]
                input_.append(num_row[0:9])
                output1_.append(num_row[9:10])
                output2_.append(num_row[10:11])
            
            
        
        #trX = np.linspace(-1, 1, 101)
        #trY1 = np.arange(4).reshape(2,2)
        my_X = np.array(input_[0:data_lenth])
        my_Y1 = np.array(output1_[0:data_lenth])
        my_Y2 = np.array(output2_[0:data_lenth])
        
        self.raw_input = my_X
        self.raw_output_1 = my_Y1
        self.raw_output_2 = my_Y2
        
        self.input_mean = np.mean(my_X, axis=0)
        self.input_max = np.max(my_X, axis=0)
        self.input_min = np.min(my_X, axis=0)
        print(self.input_mean)
        print(self.input_max)
        print(self.input_min)
        scaler_x = StandardScaler().fit(my_X)
        scaler_y1 = StandardScaler().fit(my_Y1)
        scaler_y2 = StandardScaler().fit(my_Y2)
        self.scaler_x = scaler_x
        self.scaler_y1 = scaler_y1
        self.scaler_y2 = scaler_y2
        
        
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
          W1 = sess.run(w)
          
        
          # 输出 b 的值 
          B1 = sess.run(biases)
          self.B1 = B1
        
        #test_input = np.transpose(np.transpose(np.array([100,5000,4000])))
        test_input = self.input_mean
        W1 = np.transpose(W1)
        self.W1 = W1
        
        print("************* testing response_time_rnd ****************")
        print("W1:")
        print(W1)
        print("B1:")
        print(B1)
        print("testing data: ")
        print(test_input)
        test_X = scaler_x.transform(test_input)
        test_output = np.dot(test_X,W1)+B1
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
          W2 = sess.run(w)
          # 输出 b 的值 
          B2 = sess.run(biases)
          self.B2 = B2
          
        W2 = np.transpose(W2)
        self.W2 = W2
        print("************* testing  response_time_seq ****************")
        print("W2:")
        print(W2)
        print("B2:")
        print(B2)
        print("testing data: ")
        print(test_input)
        test_X = scaler_x.transform(test_input)
        test_output = np.dot(test_X,W2)+B2
        print("result data(response_time_seq): ")
        print(scaler_y2.inverse_transform(test_output))
        
        self.train_label.setText("W1:"+str(np.transpose(W1))+"\nB1:"+str(B1)+"\nW2:"+str(np.transpose(W2))+"\nB2:"+str(B2))





if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
