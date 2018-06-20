#!/usr/bin/python3.6
# -*- coding:utf-8 -*-
__author__ = 'david'

import os
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

xl_dir = 'F://pythonwork//shengchenghanzi/makehanzi_28_28/chinese/xldatav2'
# ---训练----
ren1_xl = []
bu2_xl = []
feng3_xl = []
wu4_xl = []
hua5_xl = []
chun6_xl=[]
yi7_xl=[]
shan8_xl=[]
tian9_xl=[]
yue10_xl=[]

# 生成ren1训练集
for file in os.listdir(xl_dir + '/ren1_xl'):
    ren1_xl.append(xl_dir + '/ren1_xl' + '/' + file)
#为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(ren1_xl[0])
dim=img0.shape
image_ren1_xl=np.zeros([len(ren1_xl),dim[0],dim[1]])#存储为矩阵形式
label_ren1_xl=np.zeros([len(ren1_xl),10]).astype('int')
label_ren1_xl[:,0]=1
for k in range(len(ren1_xl)):
    img_temp = imread(ren1_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh1=img_dtd/np.amax(img_dtd)   #归一化
    image_ren1_xl[k] = img_gyh1   #将28*28图像转后为行向量
# 生成bu2训练集
for file in os.listdir(xl_dir + '/bu2_xl'):
    bu2_xl.append(xl_dir + '/bu2_xl' + '/' + file)
img0 = imread(bu2_xl[0])
dim=img0.shape
image_bu2_xl=np.zeros([len(bu2_xl),dim[0],dim[1]])
label_bu2_xl=np.zeros([len(bu2_xl),10]).astype('int')
label_bu2_xl[:,1]=1
for k in range(len(bu2_xl)):
    img_temp = imread(bu2_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh2=img_dtd/np.amax(img_dtd)   #归一化
    image_bu2_xl[k] = img_gyh2

# 生成feng3训练集
for file in os.listdir(xl_dir + '/feng3_xl'):
    feng3_xl.append(xl_dir + '/feng3_xl' + '/' + file)
img0 = imread(feng3_xl[0])
dim=img0.shape
image_feng3_xl=np.zeros([len(feng3_xl),dim[0],dim[1]])
label_feng3_xl=np.zeros([len(feng3_xl),10]).astype('int')
label_feng3_xl[:,2]=1
for k in range(len(feng3_xl)):
    img_temp = imread(feng3_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh3=img_dtd/np.amax(img_dtd)   #归一化
    image_feng3_xl[k] = img_gyh3

# 生成wu4训练集
for file in os.listdir(xl_dir + '/wu4_xl'):
    wu4_xl.append(xl_dir + '/wu4_xl' + '/' + file)
img0 = imread(wu4_xl[0])
dim=img0.shape
image_wu4_xl=np.zeros([len(wu4_xl),dim[0],dim[1]])
label_wu4_xl=np.zeros([len(wu4_xl),10]).astype('int')
label_wu4_xl[:,3]=1
for k in range(len(wu4_xl)):
    img_temp = imread(wu4_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh4=img_dtd/np.amax(img_dtd)   #归一化
    image_wu4_xl[k] = img_gyh4

# 生成hua5训练集
for file in os.listdir(xl_dir + '/hua5_xl'):
    hua5_xl.append(xl_dir + '/hua5_xl' + '/' + file)
img0 = imread(hua5_xl[0])
dim=img0.shape
image_hua5_xl=np.zeros([len(hua5_xl),dim[0],dim[1]])
label_hua5_xl=np.zeros([len(hua5_xl),10]).astype('int')
label_hua5_xl[:,4]=1
for k in range(len(hua5_xl)):
    img_temp = imread(hua5_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh5=img_dtd/np.amax(img_dtd)   #归一化
    image_hua5_xl[k] = img_gyh5

# 生成chun6训练集
for file in os.listdir(xl_dir + '/chun6_xl'):
    chun6_xl.append(xl_dir + '/chun6_xl' + '/' + file)
img0 = imread(chun6_xl[0])
dim=img0.shape
image_chun6_xl=np.zeros([len(chun6_xl),dim[0],dim[1]])
label_chun6_xl=np.zeros([len(chun6_xl),10]).astype('int')
label_chun6_xl[:,5]=1
for k in range(len(chun6_xl)):
    img_temp = imread(chun6_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh6=img_dtd/np.amax(img_dtd)   #归一化
    image_chun6_xl[k] = img_gyh6

# 生成yi7训练集
for file in os.listdir(xl_dir + '/yi7_xl'):
    yi7_xl.append(xl_dir + '/yi7_xl' + '/' + file)
img0 = imread(yi7_xl[0])
dim=img0.shape
image_yi7_xl=np.zeros([len(yi7_xl),dim[0],dim[1]])
label_yi7_xl=np.zeros([len(yi7_xl),10]).astype('int')
label_yi7_xl[:,6]=1
for k in range(len(yi7_xl)):
    img_temp = imread(yi7_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh7=img_dtd/np.amax(img_dtd)   #归一化
    image_yi7_xl[k] = img_gyh7

# 生成shan8训练集
for file in os.listdir(xl_dir + '/shan8_xl'):
    shan8_xl.append(xl_dir + '/shan8_xl' + '/' + file)
img0 = imread(shan8_xl[0])
dim=img0.shape
image_shan8_xl=np.zeros([len(shan8_xl),dim[0],dim[1]])
label_shan8_xl=np.zeros([len(shan8_xl),10]).astype('int')
label_shan8_xl[:,7]=1
for k in range(len(shan8_xl)):
    img_temp = imread(shan8_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh8=img_dtd/np.amax(img_dtd)   #归一化
    image_shan8_xl[k] = img_gyh8

# 生成tian9训练集
for file in os.listdir(xl_dir + '/tian9_xl'):
    tian9_xl.append(xl_dir + '/tian9_xl' + '/' + file)
img0 = imread(tian9_xl[0])
dim=img0.shape
image_tian9_xl=np.zeros([len(tian9_xl),dim[0],dim[1]])
label_tian9_xl=np.zeros([len(tian9_xl),10]).astype('int')
label_tian9_xl[:,8]=1
for k in range(len(tian9_xl)):
    img_temp = imread(tian9_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh9=img_dtd/np.amax(img_dtd)   #归一化
    image_tian9_xl[k] = img_gyh9

# 生成yue10训练集
for file in os.listdir(xl_dir + '/yue10_xl'):
    yue10_xl.append(xl_dir + '/yue10_xl' + '/' + file)
img0 = imread(yue10_xl[0])
dim=img0.shape
image_yue10_xl=np.zeros([len(yue10_xl),dim[0],dim[1]])
label_yue10_xl=np.zeros([len(yue10_xl),10]).astype('int')
label_yue10_xl[:,9]=1
for k in range(len(yue10_xl)):
    img_temp = imread(yue10_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh10=img_dtd/np.amax(img_dtd)   #归一化
    image_yue10_xl[k] = img_gyh10

np.savez('ziti_xldata_10_28_28.npz',image_ren1_xl=image_ren1_xl,label_ren1_xl=label_ren1_xl, \
         image_bu2_xl=image_bu2_xl, label_bu2_xl=label_bu2_xl, \
         image_feng3_xl=image_feng3_xl, label_feng3_xl=label_feng3_xl, \
         image_wu4_xl=image_wu4_xl, label_wu4_xl=label_wu4_xl, \
         image_hua5_xl=image_hua5_xl, label_hua5_xl=label_hua5_xl, \
         image_chun6_xl=image_chun6_xl, label_chun6_xl=label_chun6_xl, \
         image_yi7_xl=image_yi7_xl, label_yi7_xl=label_yi7_xl, \
         image_shan8_xl=image_shan8_xl, label_shan8_xl=label_shan8_xl, \
         image_tian9_xl=image_tian9_xl, label_tian9_xl=label_tian9_xl, \
         image_yue10_xl=image_yue10_xl, label_yue10_xl=label_yue10_xl)






