#!/usr/bin/python3.6
# -*- coding:utf-8 -*-
__author__ = 'david'

import os
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

cs_dir = 'F://pythonwork//shengchenghanzi//makehanzi_28_28//chinese//csdatav2'
# ---训练----
ren1_cs = []
bu2_cs = []
feng3_cs = []
wu4_cs = []
hua5_cs = []
chun6_cs=[]
yi7_cs=[]
shan8_cs=[]
tian9_cs=[]
yue10_cs=[]

# 生成ren1训练集
for file in os.listdir(cs_dir + '/ren1_cs'):
    ren1_cs.append(cs_dir + '/ren1_cs' + '/' + file)
img0 = imread(ren1_cs[0])
dim=img0.shape
image_ren1_cs=np.zeros([len(ren1_cs),dim[0],dim[1]])
label_ren1_cs=np.zeros([len(ren1_cs),10]).astype('int')
label_ren1_cs[:,0]=1
for k in range(len(ren1_cs)):
    img_temp = imread(ren1_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh1=img_dtd/np.amax(img_dtd)   #归一化
    image_ren1_cs[k] = img_gyh1
# 生成bu2训练集
for file in os.listdir(cs_dir + '/bu2_cs'):
    bu2_cs.append(cs_dir + '/bu2_cs' + '/' + file)
img0 = imread(bu2_cs[0])
dim=img0.shape
image_bu2_cs=np.zeros([len(bu2_cs),dim[0],dim[1]])
label_bu2_cs=np.zeros([len(bu2_cs),10]).astype('int')
label_bu2_cs[:,1]=1
for k in range(len(bu2_cs)):
    img_temp = imread(bu2_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh2=img_dtd/np.amax(img_dtd)   #归一化
    image_bu2_cs[k] = img_gyh2

# 生成feng3训练集
for file in os.listdir(cs_dir + '/feng3_cs'):
    feng3_cs.append(cs_dir + '/feng3_cs' + '/' + file)
img0 = imread(feng3_cs[0])
dim=img0.shape
image_feng3_cs=np.zeros([len(feng3_cs),dim[0],dim[1]])
label_feng3_cs=np.zeros([len(feng3_cs),10]).astype('int')
label_feng3_cs[:,2]=1
for k in range(len(feng3_cs)):
    img_temp = imread(feng3_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh3=img_dtd/np.amax(img_dtd)   #归一化
    image_feng3_cs[k] = img_gyh3

# 生成wu4训练集
for file in os.listdir(cs_dir + '/wu4_cs'):
    wu4_cs.append(cs_dir + '/wu4_cs' + '/' + file)
img0 = imread(wu4_cs[0])
dim=img0.shape
image_wu4_cs=np.zeros([len(wu4_cs),dim[0],dim[1]])
label_wu4_cs=np.zeros([len(wu4_cs),10]).astype('int')
label_wu4_cs[:,3]=1
for k in range(len(wu4_cs)):
    img_temp = imread(wu4_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh4=img_dtd/np.amax(img_dtd)   #归一化
    image_wu4_cs[k] = img_gyh4

# 生成hua5训练集
for file in os.listdir(cs_dir + '/hua5_cs'):
    hua5_cs.append(cs_dir + '/hua5_cs' + '/' + file)
img0 = imread(hua5_cs[0])
dim=img0.shape
image_hua5_cs=np.zeros([len(hua5_cs),dim[0],dim[1]])
label_hua5_cs=np.zeros([len(hua5_cs),10]).astype('int')
label_hua5_cs[:,4]=1
for k in range(len(hua5_cs)):
    img_temp = imread(hua5_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh5=img_dtd/np.amax(img_dtd)   #归一化
    image_hua5_cs[k] = img_gyh5

# 生成chun6训练集
for file in os.listdir(cs_dir + '/chun6_cs'):
    chun6_cs.append(cs_dir + '/chun6_cs' + '/' + file)
img0 = imread(chun6_cs[0])
dim=img0.shape
image_chun6_cs=np.zeros([len(chun6_cs),dim[0],dim[1]])
label_chun6_cs=np.zeros([len(chun6_cs),10]).astype('int')
label_chun6_cs[:,5]=1
for k in range(len(chun6_cs)):
    img_temp = imread(chun6_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh6=img_dtd/np.amax(img_dtd)   #归一化
    image_chun6_cs[k] = img_gyh6

# 生成yi7训练集
for file in os.listdir(cs_dir + '/yi7_cs'):
    yi7_cs.append(cs_dir + '/yi7_cs' + '/' + file)
img0 = imread(yi7_cs[0])
dim=img0.shape
image_yi7_cs=np.zeros([len(yi7_cs),dim[0],dim[1]])
label_yi7_cs=np.zeros([len(yi7_cs),10]).astype('int')
label_yi7_cs[:,6]=1
for k in range(len(yi7_cs)):
    img_temp = imread(yi7_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh7=img_dtd/np.amax(img_dtd)   #归一化
    image_yi7_cs[k] = img_gyh7

# 生成shan8训练集
for file in os.listdir(cs_dir + '/shan8_cs'):
    shan8_cs.append(cs_dir + '/shan8_cs' + '/' + file)
img0 = imread(shan8_cs[0])
dim=img0.shape
image_shan8_cs=np.zeros([len(shan8_cs),dim[0],dim[1]])
label_shan8_cs=np.zeros([len(shan8_cs),10]).astype('int')
label_shan8_cs[:,7]=1
for k in range(len(shan8_cs)):
    img_temp = imread(shan8_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh8=img_dtd/np.amax(img_dtd)   #归一化
    image_shan8_cs[k] = img_gyh8

# 生成tian9训练集
for file in os.listdir(cs_dir + '/tian9_cs'):
    tian9_cs.append(cs_dir + '/tian9_cs' + '/' + file)
img0 = imread(tian9_cs[0])
dim=img0.shape
image_tian9_cs=np.zeros([len(tian9_cs),dim[0],dim[1]])
label_tian9_cs=np.zeros([len(tian9_cs),10]).astype('int')
label_tian9_cs[:,8]=1
for k in range(len(tian9_cs)):
    img_temp = imread(tian9_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh9=img_dtd/np.amax(img_dtd)   #归一化
    image_tian9_cs[k] = img_gyh9

# 生成yue10训练集
for file in os.listdir(cs_dir + '/yue10_cs'):
    yue10_cs.append(cs_dir + '/yue10_cs' + '/' + file)
img0 = imread(yue10_cs[0])
dim=img0.shape
image_yue10_cs=np.zeros([len(yue10_cs),dim[0],dim[1]])
label_yue10_cs=np.zeros([len(yue10_cs),10]).astype('int')
label_yue10_cs[:,9]=1
for k in range(len(yue10_cs)):
    img_temp = imread(yue10_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh10=img_dtd/np.amax(img_dtd)   #归一化
    image_yue10_cs[k] = img_gyh10

np.savez('ziti_csdata_10_28_28.npz',image_ren1_cs=image_ren1_cs,label_ren1_cs=label_ren1_cs, \
         image_bu2_cs=image_bu2_cs, label_bu2_cs=label_bu2_cs, \
         image_feng3_cs=image_feng3_cs, label_feng3_cs=label_feng3_cs, \
         image_wu4_cs=image_wu4_cs, label_wu4_cs=label_wu4_cs, \
         image_hua5_cs=image_hua5_cs, label_hua5_cs=label_hua5_cs, \
         image_chun6_cs=image_chun6_cs, label_chun6_cs=label_chun6_cs, \
         image_yi7_cs=image_yi7_cs, label_yi7_cs=label_yi7_cs, \
         image_shan8_cs=image_shan8_cs, label_shan8_cs=label_shan8_cs, \
         image_tian9_cs=image_tian9_cs, label_tian9_cs=label_tian9_cs, \
         image_yue10_cs=image_yue10_cs, label_yue10_cs=label_yue10_cs)






