#!/usr/bin/python3.6
# -*- coding:utf-8 -*-
__author__ = 'david'
#生成20个测试字体，每个字体包含[1,28*28]样本和20位标签
import os
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
N=20  #待识别的汉字个数
cs_dir = 'F://pythonwork//shengchenghanzi//makehanzi_28_28//chinese//csdatav2'
# ---测试----
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
shi11_cs=[]
yun12_cs=[]
lai13_cs=[]
ri14_cs=[]
ye15_cs=[]
you16_cs=[]
jiang17_cs=[]
shui18_cs=[]
chang19_cs=[]
nian20_cs=[]

# 生成ren1测试集
for file in os.listdir(cs_dir + '/ren1_cs'):
    ren1_cs.append(cs_dir + '/ren1_cs' + '/' + file)
img0 = imread(ren1_cs[0])
dim=img0.shape
image_ren1_cs=np.zeros([len(ren1_cs),dim[0],dim[1]]).reshape([len(ren1_cs),dim[0]*dim[1]])
label_ren1_cs=np.zeros([len(ren1_cs),N]).astype('int')
label_ren1_cs[:,0]=1
for k in range(len(ren1_cs)):
    img_temp = imread(ren1_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh1=img_dtd/np.amax(img_dtd)   #归一化
    image_ren1_cs[k] = img_gyh1.reshape([1,dim[0]*dim[1]])
# 生成bu2测试集
for file in os.listdir(cs_dir + '/bu2_cs'):
    bu2_cs.append(cs_dir + '/bu2_cs' + '/' + file)
img0 = imread(bu2_cs[0])
dim=img0.shape
image_bu2_cs=np.zeros([len(bu2_cs),dim[0],dim[1]]).reshape([len(bu2_cs),dim[0]*dim[1]])
label_bu2_cs=np.zeros([len(bu2_cs),N]).astype('int')
label_bu2_cs[:,1]=1
for k in range(len(bu2_cs)):
    img_temp = imread(bu2_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh2=img_dtd/np.amax(img_dtd)   #归一化
    image_bu2_cs[k] = img_gyh2.reshape([1,dim[0]*dim[1]])

# 生成feng3测试集
for file in os.listdir(cs_dir + '/feng3_cs'):
    feng3_cs.append(cs_dir + '/feng3_cs' + '/' + file)
img0 = imread(feng3_cs[0])
dim=img0.shape
image_feng3_cs=np.zeros([len(feng3_cs),dim[0],dim[1]]).reshape([len(feng3_cs),dim[0]*dim[1]])
label_feng3_cs=np.zeros([len(feng3_cs),N]).astype('int')
label_feng3_cs[:,2]=1
for k in range(len(feng3_cs)):
    img_temp = imread(feng3_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh3=img_dtd/np.amax(img_dtd)   #归一化
    image_feng3_cs[k] = img_gyh3.reshape([1,dim[0]*dim[1]])

# 生成wu4测试集
for file in os.listdir(cs_dir + '/wu4_cs'):
    wu4_cs.append(cs_dir + '/wu4_cs' + '/' + file)
img0 = imread(wu4_cs[0])
dim=img0.shape
image_wu4_cs=np.zeros([len(wu4_cs),dim[0],dim[1]]).reshape([len(wu4_cs),dim[0]*dim[1]])
label_wu4_cs=np.zeros([len(wu4_cs),N]).astype('int')
label_wu4_cs[:,3]=1
for k in range(len(wu4_cs)):
    img_temp = imread(wu4_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh4=img_dtd/np.amax(img_dtd)   #归一化
    image_wu4_cs[k] = img_gyh4.reshape([1,dim[0]*dim[1]])

# 生成hua5测试集
for file in os.listdir(cs_dir + '/hua5_cs'):
    hua5_cs.append(cs_dir + '/hua5_cs' + '/' + file)
img0 = imread(hua5_cs[0])
dim=img0.shape
image_hua5_cs=np.zeros([len(hua5_cs),dim[0],dim[1]]).reshape([len(hua5_cs),dim[0]*dim[1]])
label_hua5_cs=np.zeros([len(hua5_cs),N]).astype('int')
label_hua5_cs[:,4]=1
for k in range(len(hua5_cs)):
    img_temp = imread(hua5_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh5=img_dtd/np.amax(img_dtd)   #归一化
    image_hua5_cs[k] = img_gyh5.reshape([1,dim[0]*dim[1]])

# 生成chun6测试集
for file in os.listdir(cs_dir + '/chun6_cs'):
    chun6_cs.append(cs_dir + '/chun6_cs' + '/' + file)
img0 = imread(chun6_cs[0])
dim=img0.shape
image_chun6_cs=np.zeros([len(chun6_cs),dim[0],dim[1]]).reshape([len(chun6_cs),dim[0]*dim[1]])
label_chun6_cs=np.zeros([len(chun6_cs),N]).astype('int')
label_chun6_cs[:,5]=1
for k in range(len(chun6_cs)):
    img_temp = imread(chun6_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh6=img_dtd/np.amax(img_dtd)   #归一化
    image_chun6_cs[k] = img_gyh6.reshape([1,dim[0]*dim[1]])

# 生成yi7测试集
for file in os.listdir(cs_dir + '/yi7_cs'):
    yi7_cs.append(cs_dir + '/yi7_cs' + '/' + file)
img0 = imread(yi7_cs[0])
dim=img0.shape
image_yi7_cs=np.zeros([len(yi7_cs),dim[0],dim[1]]).reshape([len(yi7_cs),dim[0]*dim[1]])
label_yi7_cs=np.zeros([len(yi7_cs),N]).astype('int')
label_yi7_cs[:,6]=1
for k in range(len(yi7_cs)):
    img_temp = imread(yi7_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh7=img_dtd/np.amax(img_dtd)   #归一化
    image_yi7_cs[k] = img_gyh7.reshape([1,dim[0]*dim[1]])

# 生成shan8测试集
for file in os.listdir(cs_dir + '/shan8_cs'):
    shan8_cs.append(cs_dir + '/shan8_cs' + '/' + file)
img0 = imread(shan8_cs[0])
dim=img0.shape
image_shan8_cs=np.zeros([len(shan8_cs),dim[0],dim[1]]).reshape([len(shan8_cs),dim[0]*dim[1]])
label_shan8_cs=np.zeros([len(shan8_cs),N]).astype('int')
label_shan8_cs[:,7]=1
for k in range(len(shan8_cs)):
    img_temp = imread(shan8_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh8=img_dtd/np.amax(img_dtd)   #归一化
    image_shan8_cs[k] = img_gyh8.reshape([1,dim[0]*dim[1]])

# 生成tian9测试集
for file in os.listdir(cs_dir + '/tian9_cs'):
    tian9_cs.append(cs_dir + '/tian9_cs' + '/' + file)
img0 = imread(tian9_cs[0])
dim=img0.shape
image_tian9_cs=np.zeros([len(tian9_cs),dim[0],dim[1]]).reshape([len(tian9_cs),dim[0]*dim[1]])
label_tian9_cs=np.zeros([len(tian9_cs),N]).astype('int')
label_tian9_cs[:,8]=1
for k in range(len(tian9_cs)):
    img_temp = imread(tian9_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh9=img_dtd/np.amax(img_dtd)   #归一化
    image_tian9_cs[k] = img_gyh9.reshape([1,dim[0]*dim[1]])

# 生成yue10测试集
for file in os.listdir(cs_dir + '/yue10_cs'):
    yue10_cs.append(cs_dir + '/yue10_cs' + '/' + file)
img0 = imread(yue10_cs[0])
dim=img0.shape
image_yue10_cs=np.zeros([len(yue10_cs),dim[0],dim[1]]).reshape([len(yue10_cs),dim[0]*dim[1]])
label_yue10_cs=np.zeros([len(yue10_cs),N]).astype('int')
label_yue10_cs[:,9]=1
for k in range(len(yue10_cs)):
    img_temp = imread(yue10_cs[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh10=img_dtd/np.amax(img_dtd)   #归一化
    image_yue10_cs[k] = img_gyh10.reshape([1,dim[0]*dim[1]])

# 生成shi11测试集
for file in os.listdir(cs_dir + '/shi11_cs'):
    shi11_cs.append(cs_dir + '/shi11_cs' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(shi11_cs[0])
dim = img0.shape
image_shi11_cs = np.zeros([len(shi11_cs), dim[0], dim[1]]).reshape([len(shi11_cs), dim[0] * dim[1]])  # 存储为行向量形式
label_shi11_cs = np.zeros([len(shi11_cs), N]).astype('int')
label_shi11_cs[:, 10] = 1
for k in range(len(shi11_cs)):
    img_temp = imread(shi11_cs[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_shi11_cs[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量

# 生成yun12测试集
for file in os.listdir(cs_dir + '/yun12_cs'):
    yun12_cs.append(cs_dir + '/yun12_cs' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(yun12_cs[0])
dim = img0.shape
image_yun12_cs = np.zeros([len(yun12_cs), dim[0], dim[1]]).reshape([len(yun12_cs), dim[0] * dim[1]])  # 存储为行向量形式
label_yun12_cs = np.zeros([len(yun12_cs), N]).astype('int')
label_yun12_cs[:, 11] = 1
for k in range(len(yun12_cs)):
    img_temp = imread(yun12_cs[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_yun12_cs[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量

# 生成lai13测试集
for file in os.listdir(cs_dir + '/lai13_cs'):
    lai13_cs.append(cs_dir + '/lai13_cs' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(lai13_cs[0])
dim = img0.shape
image_lai13_cs = np.zeros([len(lai13_cs), dim[0], dim[1]]).reshape([len(lai13_cs), dim[0] * dim[1]])  # 存储为行向量形式
label_lai13_cs = np.zeros([len(lai13_cs), N]).astype('int')
label_lai13_cs[:, 12] = 1
for k in range(len(lai13_cs)):
    img_temp = imread(lai13_cs[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_lai13_cs[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量

# 生成ri14测试集
for file in os.listdir(cs_dir + '/ri14_cs'):
    ri14_cs.append(cs_dir + '/ri14_cs' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(ri14_cs[0])
dim = img0.shape
image_ri14_cs = np.zeros([len(ri14_cs), dim[0], dim[1]]).reshape([len(ri14_cs), dim[0] * dim[1]])  # 存储为行向量形式
label_ri14_cs = np.zeros([len(ri14_cs), N]).astype('int')
label_ri14_cs[:, 13] = 1
for k in range(len(ri14_cs)):
    img_temp = imread(ri14_cs[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_ri14_cs[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量

# 生成ye15测试集
for file in os.listdir(cs_dir + '/ye15_cs'):
    ye15_cs.append(cs_dir + '/ye15_cs' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(ye15_cs[0])
dim = img0.shape
image_ye15_cs = np.zeros([len(ye15_cs), dim[0], dim[1]]).reshape([len(ye15_cs), dim[0] * dim[1]])  # 存储为行向量形式
label_ye15_cs = np.zeros([len(ye15_cs), N]).astype('int')
label_ye15_cs[:, 14] = 1
for k in range(len(ye15_cs)):
    img_temp = imread(ye15_cs[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_ye15_cs[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量

# 生成you16测试集
for file in os.listdir(cs_dir + '/you16_cs'):
    you16_cs.append(cs_dir + '/you16_cs' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(you16_cs[0])
dim = img0.shape
image_you16_cs = np.zeros([len(you16_cs), dim[0], dim[1]]).reshape([len(you16_cs), dim[0] * dim[1]])  # 存储为行向量形式
label_you16_cs = np.zeros([len(you16_cs), N]).astype('int')
label_you16_cs[:, 15] = 1
for k in range(len(you16_cs)):
    img_temp = imread(you16_cs[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_you16_cs[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量

# 生成jiang17测试集
for file in os.listdir(cs_dir + '/jiang17_cs'):
    jiang17_cs.append(cs_dir + '/jiang17_cs' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(jiang17_cs[0])
dim = img0.shape
image_jiang17_cs = np.zeros([len(jiang17_cs), dim[0], dim[1]]).reshape([len(jiang17_cs), dim[0] * dim[1]])  # 存储为行向量形式
label_jiang17_cs = np.zeros([len(jiang17_cs), N]).astype('int')
label_jiang17_cs[:, 16] = 1
for k in range(len(jiang17_cs)):
    img_temp = imread(jiang17_cs[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_jiang17_cs[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量

# 生成shui18测试集
for file in os.listdir(cs_dir + '/shui18_cs'):
    shui18_cs.append(cs_dir + '/shui18_cs' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(shui18_cs[0])
dim = img0.shape
image_shui18_cs = np.zeros([len(shui18_cs), dim[0], dim[1]]).reshape([len(shui18_cs), dim[0] * dim[1]])  # 存储为行向量形式
label_shui18_cs = np.zeros([len(shui18_cs), N]).astype('int')
label_shui18_cs[:, 17] = 1
for k in range(len(shui18_cs)):
    img_temp = imread(shui18_cs[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_shui18_cs[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量

# 生成chang19测试集
for file in os.listdir(cs_dir + '/chang19_cs'):
    chang19_cs.append(cs_dir + '/chang19_cs' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(chang19_cs[0])
dim = img0.shape
image_chang19_cs = np.zeros([len(chang19_cs), dim[0], dim[1]]).reshape([len(chang19_cs), dim[0] * dim[1]])  # 存储为行向量形式
label_chang19_cs = np.zeros([len(chang19_cs), N]).astype('int')
label_chang19_cs[:, 18] = 1
for k in range(len(chang19_cs)):
    img_temp = imread(chang19_cs[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_chang19_cs[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量

# 生成nian20测试集
for file in os.listdir(cs_dir + '/nian20_cs'):
    nian20_cs.append(cs_dir + '/nian20_cs' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(nian20_cs[0])
dim = img0.shape
image_nian20_cs = np.zeros([len(nian20_cs), dim[0], dim[1]]).reshape([len(nian20_cs), dim[0] * dim[1]])  # 存储为行向量形式
label_nian20_cs = np.zeros([len(nian20_cs), N]).astype('int')
label_nian20_cs[:, 19] = 1
for k in range(len(nian20_cs)):
    img_temp = imread(nian20_cs[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_nian20_cs[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量

np.savez('ziti_csdata_20_1_784.npz',image_ren1_cs=image_ren1_cs,label_ren1_cs=label_ren1_cs, \
            image_bu2_cs=image_bu2_cs, label_bu2_cs=label_bu2_cs, \
            image_feng3_cs=image_feng3_cs, label_feng3_cs=label_feng3_cs, \
            image_wu4_cs=image_wu4_cs, label_wu4_cs=label_wu4_cs, \
            image_hua5_cs=image_hua5_cs, label_hua5_cs=label_hua5_cs, \
            image_chun6_cs=image_chun6_cs, label_chun6_cs=label_chun6_cs, \
            image_yi7_cs=image_yi7_cs, label_yi7_cs=label_yi7_cs, \
            image_shan8_cs=image_shan8_cs, label_shan8_cs=label_shan8_cs, \
            image_tian9_cs=image_tian9_cs, label_tian9_cs=label_tian9_cs, \
            image_yue10_cs=image_yue10_cs, label_yue10_cs=label_yue10_cs, \
            image_shi11_cs=image_shi11_cs, label_shi11_cs=label_shi11_cs, \
            image_yun12_cs=image_yun12_cs, label_yun12_cs=label_yun12_cs, \
            image_lai13_cs=image_lai13_cs, label_lai13_cs=label_lai13_cs, \
            image_ri14_cs=image_ri14_cs, label_ri14_cs=label_ri14_cs, \
            image_ye15_cs=image_ye15_cs, label_ye15_cs=label_ye15_cs, \
            image_you16_cs=image_you16_cs, label_you16_cs=label_you16_cs, \
            image_jiang17_cs=image_jiang17_cs, label_jiang17_cs=label_jiang17_cs, \
            image_shui18_cs=image_shui18_cs, label_shui18_cs=label_shui18_cs, \
            image_chang19_cs=image_chang19_cs, label_chang19_cs=label_chang19_cs, \
            image_nian20_cs=image_nian20_cs, label_nian20_cs=label_nian20_cs)






