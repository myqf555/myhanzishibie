#!/usr/bin/python3.6
# -*- coding:utf-8 -*-
__author__ = 'david'

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import matplotlib.pyplot as plt
import numpy as np
# 防止图中出现乱码
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

sess=tf.Session()
# 加载 结构，即 模型参数 变量等
new_saver = tf.train.import_meta_graph("F://pythonwork//shengchenghanzi//makehanzi_28_28//model_cnn10_28_28//model_cnn10_28_28-54000.meta")
print("ModelV construct!!")
all_vars = tf.trainable_variables()
for v in all_vars:
    print(v.name)
#     print(v.name,v.eval(sess))  # v 都还未初始化，不能求值
# 加载模型参数变量的 值
new_saver.restore(sess,tf.train.latest_checkpoint('F://pythonwork//shengchenghanzi//makehanzi_28_28//model_cnn10_28_28'))
print("ModelV restored!!")
# all_vars = tf.trainable_variables()
# for v in all_vars:
#     print(v.name,v.eval(sess))


reader = pywrap_tensorflow.NewCheckpointReader("F://pythonwork/shengchenghanzi/makehanzi_28_28/model_cnn10_28_28/model_cnn10_28_28-54000")


HB=np.load('ziti_xldata_hebing_10_1_784.npz')
data_xl= HB['data_xl']
dic={'0':"人","1":"不","2":"风","3":"无","4":"花","5":"春","6":"一","7":"山","8":"天","9":"月"}
W_image = 28
H_image = 28
np.random.shuffle(data_xl)
xl_image = data_xl[:, 0:W_image * H_image].reshape([-1,28,28,1])
xl_label = data_xl[:, W_image*H_image:W_image*H_image+10]

image = xl_image[0].reshape([28,28])
label = xl_label[0]
word_name=dic[str(np.argmax(label))]

# 绘制原始图像
fig0 = plt.figure(0)#,figsize=(28,28),dpi=4)#figsize=(28,28),dpi=4)
ax0 = fig0.add_subplot(111)
ax0.imshow(image,cmap=plt.cm.gray)
ax0.set_title('"'+word_name+'"'+'原始图像')
plt.axis('off')
plt.show()


W_conv1 = reader.get_tensor('W_conv1')
W_conv1 = tf.cast(W_conv1,dtype = "float32")
b_conv1 = reader.get_tensor('b_conv1')
x=image.reshape([1,28,28,1])
x = tf.cast(x,dtype="float32")

xw=tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.relu(xw + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# 绘制卷积1输出图像
fig1 = plt.figure(1)#,figsize=(14,14),dpi=4)#figsize=(28,28),dpi=4)
image_dim=h_conv1.shape[3]
for k in range(image_dim):
    ax1 = fig1.add_subplot(1,2,k+1)
    h_conv1_image=sess.run(h_conv1[:,:,:,k]).reshape([28,28])
    ax1.imshow(h_conv1_image,cmap=plt.cm.gray)
    plt.title('"'+ word_name+ '"'+' 卷积层1 通道%d'%(k+1))
    plt.axis('off')
plt.show()

# #绘制池化1层图像
# fig2 = plt.figure(2,figsize=(28,28),dpi=4)#,figsize=(14,14),dpi=4)#figsize=(28,28),dpi=4)
# image_dim=2
# for k in range(image_dim):
#     ax2 = fig2.add_subplot(1,2,k+1)
#     h_pool1_temp = sess.run(h_pool1).reshape([14,14,2])
#     print(h_pool1_temp.shape)
#     h_pool1_image = h_pool1_temp[:,:,k].reshape([14,14])
#     ax2.imshow(h_pool1_image,cmap=plt.cm.gray)
#     plt.title('池化层1第%d个通道输出：'%(k+1)+word_name)
# plt.show()
W_conv2 = reader.get_tensor('W_conv2')
W_conv2 = tf.cast(W_conv2,dtype = "float32")
b_conv2 = reader.get_tensor('b_conv2')

xw1 = tf.nn.conv2d(h_pool1, W_conv2,strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu( xw1+ b_conv2)
h_pool2 = tf.nn.max_pool(h_conv1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
image_dim2=sess.run(W_conv2).shape[3]
fig3 = plt.figure(3)#,figsize=(28,28),dpi=4)
for k in range(image_dim2):
    ax3 = fig3.add_subplot(4,image_dim2/4,k+1)
    h_conv2_image=sess.run(h_conv2[:,:,:,k]).reshape([14,14])
    ax3.imshow(h_conv2_image,cmap=plt.cm.gray)
    plt.title('"'+ word_name+'"'+' 卷积层2 通道%d'%(k+1))
    plt.axis('off')
plt.show()



sess.close()