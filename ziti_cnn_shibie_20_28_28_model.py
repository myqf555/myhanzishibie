#!/usr/bin/python3.6
# -*- coding:utf-8 -*-
__author__ = 'david'
# 利用卷积神经网络识别10个汉字字体
import numpy as np
import tensorflow as tf
# 防止图中出现乱码
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
N=20
BATCH_SIZE = 20
Learning_Rate=0.001 #模型训练效果影响大，决定是否会进入局部
W_image=28
H_image=28
nClass=N
epoch=20
record_step=50
keep_prob = 0.9

HB=np.load('ziti_xldata_hebing_20_1_784.npz')
data_xl= HB['data_xl']
CS=np.load('ziti_csdata_hebing_20_1_784.npz')
data_cs= CS['data_cs']

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)
def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder("float", shape=[None, W_image, H_image, 1])
Y = tf.placeholder("float", shape=[None, nClass])
# [5 5 1 2]/[2]/[5 5 2 8]/[8]/[7*7*8 1024]/[1024]/[-1, 7*7*8]/[512, N] = 92% keep_prob=0.9
# [5 5 1 2]/[2]/[5 5 2 8]/[8]/[7*7*8 512]/[512]/[-1, 7*7*8]/[512, N] = 93% keep_prob=0.8
# [4 4 1 2]/[2]/[4 4 2 8]/[8]/[7*7*8 512]/[512]/[512 256]/[256]/[-1, 512]/[256, N] = 84% keep_prob=0.7
# [5 5 1 2]/[2]/[3 3 2 2]/[2]/[5 5 2 8]/[8]/[7*7*8 1024]/[1024]/[-1, 7*7*8]/[512, N] = 87%
# [5 5 1 2]/[2]/[5 5 2 2]/[2]/[5 5 2 4]/[8]/[7*7*4 1024]/[1024]/[-1, 7*7*4]/[512, N] = 85%
# [5 5 1 2]/[2]/[5 5 2 2]/[2]/[5 5 2 2]/[8]/[7*7*2 1024]/[1024]/[-1, 7*7*2]/[512, N] = 79%
# 各个卷积层的输出通道数，对学习效果影响巨大，
# 模型过于复杂，有时候会导致过拟合，因此模型结构对于训练结果很重要
# 比如本次第1层2个，第2层输出8
# 构造第1个卷积和池化层
W_conv1 = weight_variable([5, 5, 1, 2],name='W_conv1')
b_conv1 = bias_variable([2],name='b_conv1')
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_pool1_lrn = tf.nn.lrn(h_pool1,depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='h_pool1_lrn' )

# # 构造第2个卷积和池化层
# W_conv2 = weight_variable([5, 5, 2, 2],name='W_conv2')
# b_conv2 = bias_variable([2],name='b_conv2')
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
# # h_pool2_lrn = tf.nn.lrn(h_pool2, name='h_pool2_lrn')#depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='h_pool2_lrn')

# 构造第3个卷积和池化层
W_conv3 = weight_variable([5, 5, 2, 8],name='W_conv3')
b_conv3 = bias_variable([8],name='b_conv3')
h_conv3 = tf.nn.relu(conv2d(h_pool1_lrn, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
h_pool3_lrn = tf.nn.lrn(h_pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='h_pool2_lrn')
# h_pool3 = h_conv3
#全连接层1
# 28*28-->14*14--->7*7-->4*4
W_fc1 = weight_variable([7 * 7 * 8, 512],name='W_fc1')
b_fc1 = bias_variable([512],name='b_fc1')
h_pool3_flat = tf.reshape(h_pool3_lrn, [-1, 7*7*8])#通过全连接层输出行向量 [1 7*7*64] X[7*7*64 1024]=[1 1024]
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,0.7)#keep_prob

#全连接层2
# 28*28-->14*14--->7*7-->4*4
W_fc2 = weight_variable([512, 256],name='W_fc2')
b_fc2 = bias_variable([256],name='b_fc2')
h_pool4_flat = tf.reshape(h_fc1_drop, [-1, 512])#通过全连接层输出行向量 [1 7*7*64] X[7*7*64 1024]=[1 1024]
h_fc2 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2,0.7)#keep_prob

#输出层
W_fc3 = weight_variable([256, N],name='W_fc3')
b_fc3 = bias_variable([N],'b_fc3')
y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
# 此文L2范数正则化2，减少过拟合 tf.norm(W_fc2,2)**2*0.001
# L1范数  tf.norm(W_fc2,1)*0.001 tf.norm(W_fc2,2)**2*0.005+
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=y_conv))#交叉熵
train_step = tf.train.AdamOptimizer(Learning_Rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 测试专用
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver(max_to_keep=4)

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
        xl_image = data_xl[:, 0:W_image * H_image].reshape([-1,28,28,1])
        xl_label = data_xl[:, W_image*H_image:W_image*H_image+N]
        np.random.shuffle(data_cs)
        cs_image = data_cs[:, 0:W_image * H_image].reshape([-1,28,28,1])  # 打散测试数据
        cs_label = data_cs[:, W_image*H_image:W_image*H_image+N]  # 最后N个元素表示one-hot编码
        for i in range(batch_num):
            batch_x = xl_image[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
            batch_y = xl_label[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
            sess.run(train_step, feed_dict={x: batch_x, Y: batch_y})


            if (i % record_step == 0):
                loss[num] = sess.run(cross_entropy, feed_dict={x: batch_x, Y: batch_y})
                acc_xl[num]=sess.run( accuracy, feed_dict={x: batch_x, Y: batch_y})
                acc_cs[num]=sess.run(accuracy, feed_dict={x: cs_image, Y: cs_label})
                ratio=(k * batch_num + i)/(batch_num*epoch)
                saver.save(sess, "model_cnn20_28_28/model_cnn20_28_28", global_step=k*batch_num+i)
                print("第%d次，已完成%.2f%%,Loss:%f,Training accuracy:%.2f%%,测试正确率:%.2f%%"\
                      % (k*batch_num+i,ratio*100,loss[num], acc_xl[num] * 100,acc_cs[num]*100))
                num = num + 1
    loss[num], acc_xl[num] = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x, Y: batch_y})
    acc_cs[num]=sess.run(accuracy, feed_dict={x: cs_image, Y: cs_label})
    ratio=(k * batch_num + i)/(batch_num*epoch-1)
    saver.save(sess, "model_cnn20_28_28/model_cnn20_28_28", global_step=k * batch_num + i+1)
    print("第%d次，已完成%.2f%%,Loss:%f,Training accuracy:%.2f%%,测试正确率:%.2f%%" % (k*batch_num+i+1,ratio*100,loss[num], acc_xl[num] * 100,acc_cs[num]*100))

np.savez('ziti_cnn_shibie_result_10.npz',loss=loss,acc_xl=acc_xl,acc_cs=acc_cs)

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
