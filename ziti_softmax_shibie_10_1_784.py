#!/usr/bin/python3.6
# -*- coding:utf-8 -*-
__author__ = 'david'
import numpy as np
import tensorflow as tf
# 防止图中出现乱码
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

BATCH_SIZE = 100
Learning_Rate=0.001
W_image=28
H_image=28
nClass=10
epoch=2
record_step=10

HB=np.load('ziti_xldata_hebing_10_1_784.npz')
data_xl= HB['data_xl']
CS=np.load('ziti_csdata_hebing_10_1_784.npz')
data_cs= CS['data_cs']

x = tf.placeholder("float", shape=[None, W_image*H_image])
Y = tf.placeholder("float", shape=[None, nClass])
W = tf.Variable(tf.zeros([W_image*H_image,nClass]))
b = tf.Variable(tf.zeros([nClass]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
# cross_entropy = tf.reduce_sum(tf.square(Y - y))#均方差
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=y))#交叉熵
# train_step = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(Learning_Rate).minimize(cross_entropy)
# 测试专用
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

num_xl = data_xl.shape[0]
batch_num=int(num_xl / BATCH_SIZE)
n=epoch*int(batch_num/record_step)+1
loss=np.zeros([n])
acc_xl=np.zeros([n])
acc_cs=np.zeros([n])
num=0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for k in range(epoch):
        np.random.shuffle(data_xl)
        xl_image = data_xl[:, 0:W_image * H_image]
        xl_label = data_xl[:, W_image*H_image:W_image*H_image+10]
        np.random.shuffle(data_cs)
        cs_image = data_cs[:, 0:W_image * H_image]  # 打散测试数据
        cs_label = data_cs[:, W_image*H_image:W_image*H_image+10]  # 最后10个元素表示one-hot编码
        for i in range(batch_num):
            batch_x = xl_image[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
            batch_y = xl_label[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
            sess.run(train_step, feed_dict={x: batch_x, Y: batch_y})

            if (i % record_step == 0):
                loss[num] = sess.run(cross_entropy, feed_dict={x: batch_x, Y: batch_y})
                acc_xl[num]=sess.run( accuracy, feed_dict={x: batch_x, Y: batch_y})
                acc_cs[num]=sess.run(accuracy, feed_dict={x: cs_image, Y: cs_label})
                ratio=(k * batch_num + i)/(batch_num*epoch)
                print("第%d次，已完成%.2f%%,Loss:%f,Training accuracy:%.2f%%,测试正确率:%.2f%%"\
                      % (k*batch_num+i,ratio*100,loss[num], acc_xl[num] * 100,acc_cs[num]*100))
                num = num + 1
    loss[num], acc_xl[num] = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x, Y: batch_y})
    acc_cs[num]=sess.run(accuracy, feed_dict={x: cs_image, Y: cs_label})
    ratio=(k * batch_num + i)/(batch_num*epoch)
    print("第%d次，已完成%.2f%%,Loss:%f,Training accuracy:%.2f%%,测试正确率:%.2f%%" % (k*batch_num+i,ratio*100,loss[num], acc_xl[num] * 100,acc_cs[num]*100))

fig=plt.figure()
ax=fig.add_subplot(1,2,1)
ax.plot(loss,color='black',linewidth=2,marker='.',label='损失函数')
ax.set_title('训练过程中损失函数变化情况')
ax.set_xlabel('训练次数')
ax.set_ylabel('损失函数')
ax.legend()
ax1=fig.add_subplot(1,2,2)
# ax1.plot(accuracy_xl,color='blue',linestyle=':', linewidth=2,marker='*',label='训练ACC')
# ax1.plot(accuracy_cs,color='red', linewidth=2,marker='o',label='测试ACC')
ax1.plot(acc_xl,color='blue',linestyle=':', linewidth=2,label='训练ACC')
ax1.plot(acc_cs,color='red', linewidth=2,label='测试ACC')
ax1.set_title('训练和测试识别率')
ax1.set_xlabel('训练次数')
ax1.set_ylabel('识别率ACC')
ax1.legend()
plt.show()
