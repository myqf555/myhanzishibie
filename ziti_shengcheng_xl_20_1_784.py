#!/usr/bin/python3.6
# -*- coding:utf-8 -*-
__author__ = 'david'
#生成20个训练字体，每个字体包含[1,28*28]样本和20位标签
import os
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
N=20  #待识别的汉字个数
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
shi11_xl=[]
yun12_xl=[]
lai13_xl=[]
ri14_xl=[]
ye15_xl=[]
you16_xl=[]
jiang17_xl=[]
shui18_xl=[]
chang19_xl=[]
nian20_xl=[]

# 生成ren1训练集
for file in os.listdir(xl_dir + '/ren1_xl'):
    ren1_xl.append(xl_dir + '/ren1_xl' + '/' + file)
#为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(ren1_xl[0])
dim=img0.shape
image_ren1_xl=np.zeros([len(ren1_xl),dim[0],dim[1]]).reshape([len(ren1_xl),dim[0]*dim[1]])#存储为行向量形式
label_ren1_xl=np.zeros([len(ren1_xl),N]).astype('int')
label_ren1_xl[:,0]=1
for k in range(len(ren1_xl)):
    img_temp = imread(ren1_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_ren1_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]]) #将28*28图像转后为行向量
# 生成bu2训练集
for file in os.listdir(xl_dir + '/bu2_xl'):
    bu2_xl.append(xl_dir + '/bu2_xl' + '/' + file)
img0 = imread(bu2_xl[0])
dim=img0.shape
image_bu2_xl=np.zeros([len(bu2_xl),dim[0],dim[1]]).reshape([len(bu2_xl),dim[0]*dim[1]])
label_bu2_xl=np.zeros([len(bu2_xl),N]).astype('int')
label_bu2_xl[:,1]=1
for k in range(len(bu2_xl)):
    img_temp = imread(bu2_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_bu2_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]])

# 生成feng3训练集
for file in os.listdir(xl_dir + '/feng3_xl'):
    feng3_xl.append(xl_dir + '/feng3_xl' + '/' + file)
img0 = imread(feng3_xl[0])
dim=img0.shape
image_feng3_xl=np.zeros([len(feng3_xl),dim[0],dim[1]]).reshape([len(feng3_xl),dim[0]*dim[1]])
label_feng3_xl=np.zeros([len(feng3_xl),N]).astype('int')
label_feng3_xl[:,2]=1
for k in range(len(feng3_xl)):
    img_temp = imread(feng3_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_feng3_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]])

# 生成wu4训练集
for file in os.listdir(xl_dir + '/wu4_xl'):
    wu4_xl.append(xl_dir + '/wu4_xl' + '/' + file)
img0 = imread(wu4_xl[0])
dim=img0.shape
image_wu4_xl=np.zeros([len(wu4_xl),dim[0],dim[1]]).reshape([len(wu4_xl),dim[0]*dim[1]])
label_wu4_xl=np.zeros([len(wu4_xl),N]).astype('int')
label_wu4_xl[:,3]=1
for k in range(len(wu4_xl)):
    img_temp = imread(wu4_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_wu4_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]])

# 生成hua5训练集
for file in os.listdir(xl_dir + '/hua5_xl'):
    hua5_xl.append(xl_dir + '/hua5_xl' + '/' + file)
img0 = imread(hua5_xl[0])
dim=img0.shape
image_hua5_xl=np.zeros([len(hua5_xl),dim[0],dim[1]]).reshape([len(hua5_xl),dim[0]*dim[1]])
label_hua5_xl=np.zeros([len(hua5_xl),N]).astype('int')
label_hua5_xl[:,4]=1
for k in range(len(hua5_xl)):
    img_temp = imread(hua5_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_hua5_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]])

# 生成chun6训练集
for file in os.listdir(xl_dir + '/chun6_xl'):
    chun6_xl.append(xl_dir + '/chun6_xl' + '/' + file)
img0 = imread(chun6_xl[0])
dim=img0.shape
image_chun6_xl=np.zeros([len(chun6_xl),dim[0],dim[1]]).reshape([len(chun6_xl),dim[0]*dim[1]])
label_chun6_xl=np.zeros([len(chun6_xl),N]).astype('int')
label_chun6_xl[:,5]=1
for k in range(len(chun6_xl)):
    img_temp = imread(chun6_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_chun6_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]])

# 生成yi7训练集
for file in os.listdir(xl_dir + '/yi7_xl'):
    yi7_xl.append(xl_dir + '/yi7_xl' + '/' + file)
img0 = imread(yi7_xl[0])
dim=img0.shape
image_yi7_xl=np.zeros([len(yi7_xl),dim[0],dim[1]]).reshape([len(yi7_xl),dim[0]*dim[1]])
label_yi7_xl=np.zeros([len(yi7_xl),N]).astype('int')
label_yi7_xl[:,6]=1
for k in range(len(yi7_xl)):
    img_temp = imread(yi7_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_yi7_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]])

# 生成shan8训练集
for file in os.listdir(xl_dir + '/shan8_xl'):
    shan8_xl.append(xl_dir + '/shan8_xl' + '/' + file)
img0 = imread(shan8_xl[0])
dim=img0.shape
image_shan8_xl=np.zeros([len(shan8_xl),dim[0],dim[1]]).reshape([len(shan8_xl),dim[0]*dim[1]])
label_shan8_xl=np.zeros([len(shan8_xl),N]).astype('int')
label_shan8_xl[:,7]=1
for k in range(len(shan8_xl)):
    img_temp = imread(shan8_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_shan8_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]])

# 生成tian9训练集
for file in os.listdir(xl_dir + '/tian9_xl'):
    tian9_xl.append(xl_dir + '/tian9_xl' + '/' + file)
img0 = imread(tian9_xl[0])
dim=img0.shape
image_tian9_xl=np.zeros([len(tian9_xl),dim[0],dim[1]]).reshape([len(tian9_xl),dim[0]*dim[1]])
label_tian9_xl=np.zeros([len(tian9_xl),N]).astype('int')
label_tian9_xl[:,8]=1
for k in range(len(tian9_xl)):
    img_temp = imread(tian9_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_tian9_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]])

# 生成yue10训练集
for file in os.listdir(xl_dir + '/yue10_xl'):
    yue10_xl.append(xl_dir + '/yue10_xl' + '/' + file)
img0 = imread(yue10_xl[0])
dim=img0.shape
image_yue10_xl=np.zeros([len(yue10_xl),dim[0],dim[1]]).reshape([len(yue10_xl),dim[0]*dim[1]])
label_yue10_xl=np.zeros([len(yue10_xl),N]).astype('int')
label_yue10_xl[:,9]=1
for k in range(len(yue10_xl)):
    img_temp = imread(yue10_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_yue10_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]])
    
# 生成shi11训练集
for file in os.listdir(xl_dir + '/shi11_xl'):
    shi11_xl.append(xl_dir + '/shi11_xl' + '/' + file)
#为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(shi11_xl[0])
dim=img0.shape
image_shi11_xl=np.zeros([len(shi11_xl),dim[0],dim[1]]).reshape([len(shi11_xl),dim[0]*dim[1]])#存储为行向量形式
label_shi11_xl=np.zeros([len(shi11_xl),N]).astype('int')
label_shi11_xl[:,10]=1
for k in range(len(shi11_xl)):
    img_temp = imread(shi11_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_shi11_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]]) #将28*28图像转后为行向量
    
# 生成yun12训练集
for file in os.listdir(xl_dir + '/yun12_xl'):
    yun12_xl.append(xl_dir + '/yun12_xl' + '/' + file)
#为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(yun12_xl[0])
dim=img0.shape
image_yun12_xl=np.zeros([len(yun12_xl),dim[0],dim[1]]).reshape([len(yun12_xl),dim[0]*dim[1]])#存储为行向量形式
label_yun12_xl=np.zeros([len(yun12_xl),N]).astype('int')
label_yun12_xl[:,11]=1
for k in range(len(yun12_xl)):
    img_temp = imread(yun12_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_yun12_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]]) #将28*28图像转后为行向量
    
# 生成lai13训练集
for file in os.listdir(xl_dir + '/lai13_xl'):
    lai13_xl.append(xl_dir + '/lai13_xl' + '/' + file)
#为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(lai13_xl[0])
dim=img0.shape
image_lai13_xl=np.zeros([len(lai13_xl),dim[0],dim[1]]).reshape([len(lai13_xl),dim[0]*dim[1]])#存储为行向量形式
label_lai13_xl=np.zeros([len(lai13_xl),N]).astype('int')
label_lai13_xl[:,12]=1
for k in range(len(lai13_xl)):
    img_temp = imread(lai13_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_lai13_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]]) #将28*28图像转后为行向量
    
# 生成ri14训练集
for file in os.listdir(xl_dir + '/ri14_xl'):
    ri14_xl.append(xl_dir + '/ri14_xl' + '/' + file)
#为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(ri14_xl[0])
dim=img0.shape
image_ri14_xl=np.zeros([len(ri14_xl),dim[0],dim[1]]).reshape([len(ri14_xl),dim[0]*dim[1]])#存储为行向量形式
label_ri14_xl=np.zeros([len(ri14_xl),N]).astype('int')
label_ri14_xl[:,13]=1
for k in range(len(ri14_xl)):
    img_temp = imread(ri14_xl[k])  #读取指定文件路径下文件
    img_dtd = img_temp[:,:,0]      #取单通道
    img_gyh=img_dtd/np.amax(img_dtd)   #归一化
    image_ri14_xl[k] = img_gyh.reshape([1,dim[0]*dim[1]]) #将28*28图像转后为行向量

# 生成ye15训练集
for file in os.listdir(xl_dir + '/ye15_xl'):
    ye15_xl.append(xl_dir + '/ye15_xl' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(ye15_xl[0])
dim = img0.shape
image_ye15_xl = np.zeros([len(ye15_xl), dim[0], dim[1]]).reshape([len(ye15_xl), dim[0] * dim[1]])  # 存储为行向量形式
label_ye15_xl = np.zeros([len(ye15_xl), N]).astype('int')
label_ye15_xl[:, 14] = 1
for k in range(len(ye15_xl)):
    img_temp = imread(ye15_xl[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_ye15_xl[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量

# 生成you16训练集
for file in os.listdir(xl_dir + '/you16_xl'):
    you16_xl.append(xl_dir + '/you16_xl' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(you16_xl[0])
dim = img0.shape
image_you16_xl = np.zeros([len(you16_xl), dim[0], dim[1]]).reshape([len(you16_xl), dim[0] * dim[1]])  # 存储为行向量形式
label_you16_xl = np.zeros([len(you16_xl), N]).astype('int')
label_you16_xl[:, 15] = 1
for k in range(len(you16_xl)):
    img_temp = imread(you16_xl[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_you16_xl[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量
    
# 生成jiang17训练集
for file in os.listdir(xl_dir + '/jiang17_xl'):
    jiang17_xl.append(xl_dir + '/jiang17_xl' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(jiang17_xl[0])
dim = img0.shape
image_jiang17_xl = np.zeros([len(jiang17_xl), dim[0], dim[1]]).reshape([len(jiang17_xl), dim[0] * dim[1]])  # 存储为行向量形式
label_jiang17_xl = np.zeros([len(jiang17_xl), N]).astype('int')
label_jiang17_xl[:, 16] = 1
for k in range(len(jiang17_xl)):
    img_temp = imread(jiang17_xl[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_jiang17_xl[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量

# 生成shui18训练集
for file in os.listdir(xl_dir + '/shui18_xl'):
    shui18_xl.append(xl_dir + '/shui18_xl' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(shui18_xl[0])
dim = img0.shape
image_shui18_xl = np.zeros([len(shui18_xl), dim[0], dim[1]]).reshape([len(shui18_xl), dim[0] * dim[1]])  # 存储为行向量形式
label_shui18_xl = np.zeros([len(shui18_xl), N]).astype('int')
label_shui18_xl[:, 17] = 1
for k in range(len(shui18_xl)):
    img_temp = imread(shui18_xl[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_shui18_xl[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量
    
# 生成chang19训练集
for file in os.listdir(xl_dir + '/chang19_xl'):
    chang19_xl.append(xl_dir + '/chang19_xl' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(chang19_xl[0])
dim = img0.shape
image_chang19_xl = np.zeros([len(chang19_xl), dim[0], dim[1]]).reshape([len(chang19_xl), dim[0] * dim[1]])  # 存储为行向量形式
label_chang19_xl = np.zeros([len(chang19_xl), N]).astype('int')
label_chang19_xl[:, 18] = 1
for k in range(len(chang19_xl)):
    img_temp = imread(chang19_xl[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_chang19_xl[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量
    
# 生成nian20训练集
for file in os.listdir(xl_dir + '/nian20_xl'):
    nian20_xl.append(xl_dir + '/nian20_xl' + '/' + file)
# 为了便于了解图片结构，所有首先取第一幅图片，计算器长宽值，构造存放图像的多维数组
img0 = imread(nian20_xl[0])
dim = img0.shape
image_nian20_xl = np.zeros([len(nian20_xl), dim[0], dim[1]]).reshape([len(nian20_xl), dim[0] * dim[1]])  # 存储为行向量形式
label_nian20_xl = np.zeros([len(nian20_xl), N]).astype('int')
label_nian20_xl[:, 19] = 1
for k in range(len(nian20_xl)):
    img_temp = imread(nian20_xl[k])  # 读取指定文件路径下文件
    img_dtd = img_temp[:, :, 0]  # 取单通道
    img_gyh = img_dtd / np.amax(img_dtd)  # 归一化
    image_nian20_xl[k] = img_gyh.reshape([1, dim[0] * dim[1]])  # 将28*28图像转后为行向量
    
np.savez('ziti_xldata_20_1_784.npz',image_ren1_xl=image_ren1_xl,label_ren1_xl=label_ren1_xl, \
            image_bu2_xl=image_bu2_xl, label_bu2_xl=label_bu2_xl, \
            image_feng3_xl=image_feng3_xl, label_feng3_xl=label_feng3_xl, \
            image_wu4_xl=image_wu4_xl, label_wu4_xl=label_wu4_xl, \
            image_hua5_xl=image_hua5_xl, label_hua5_xl=label_hua5_xl, \
            image_chun6_xl=image_chun6_xl, label_chun6_xl=label_chun6_xl, \
            image_yi7_xl=image_yi7_xl, label_yi7_xl=label_yi7_xl, \
            image_shan8_xl=image_shan8_xl, label_shan8_xl=label_shan8_xl, \
            image_tian9_xl=image_tian9_xl, label_tian9_xl=label_tian9_xl, \
            image_yue10_xl=image_yue10_xl, label_yue10_xl=label_yue10_xl, \
            image_shi11_xl=image_shi11_xl, label_shi11_xl=label_shi11_xl, \
            image_yun12_xl=image_yun12_xl, label_yun12_xl=label_yun12_xl,\
            image_lai13_xl=image_lai13_xl, label_lai13_xl=label_lai13_xl,\
            image_ri14_xl=image_ri14_xl, label_ri14_xl=label_ri14_xl,\
            image_ye15_xl=image_ye15_xl, label_ye15_xl=label_ye15_xl,\
            image_you16_xl=image_you16_xl, label_you16_xl=label_you16_xl,\
            image_jiang17_xl=image_jiang17_xl, label_jiang17_xl=label_jiang17_xl,\
            image_shui18_xl=image_shui18_xl, label_shui18_xl=label_shui18_xl,\
            image_chang19_xl=image_chang19_xl, label_chang19_xl=label_chang19_xl,\
            image_nian20_xl=image_nian20_xl, label_nian20_xl=label_nian20_xl)






