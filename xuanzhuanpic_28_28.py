#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'david'

import os
# import pygame
from PIL import Image
import numpy  as np
import sys
import importlib
importlib.reload(sys)

# 上面3行代码是为了防止发生“UnicodeEncodeError: ‘ascii’ codec can’t encode”错误
# =========下面设置20个汉字所在文件夹为系统当前文件夹=========================
mydir='nian20'
# yue10'/'tian9'/shan8'/'yi7 /'chun6'/'hua5'/'wu4'/'feng3'/'bu2'/'ren1'
#'nian20'/'chang19'/'shui18'/'jiang17'/'you16'/'ye15'/'ri14'/'lai13'/'yun12'/'shi11''

os.chdir('F://pythonwork//shengchenghanzi//makehanzi_28_28//chinese//'+mydir+'//')
chinese_dir = mydir+'_rotate'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)
rootdir = "F://pythonwork//shengchenghanzi//makehanzi_28_28//chinese//"+mydir+"//"
rootdir_rotate = "F://pythonwork//shengchenghanzi//makehanzi_28_28//chinese//"+mydir+"//" + chinese_dir+"//"

# ===========================================================================
list = os.listdir(rootdir)  # 列出文件夹下所有的文件
ratio_fz=0 #当前完成次数，计算总比例的分子
for num in range(0, len(list)):
    path = os.path.join(rootdir, list[num])
    if os.path.isfile(path):
        filename = os.path.abspath(path)  # 返回文件名
        file = os.path.basename(path)
        # print(filename)
        # print(file)

        # 读取图像
        im = Image.open(filename)
        # 原图像缩放为28x28
        im_resized = im.resize((28, 28))
        im2 = im_resized.convert('RGBA')
        
        # 指定逆时针旋转的角度
        angle_range = np.linspace(-45, 45, num=21)  # 产生均匀的N个数
        # print(angle_range)
        for angle_num in range(0, len(angle_range)):
            im_rotate = im2.rotate(angle_range[angle_num], expand=0)
            fff = Image.new('RGBA', im2.size, (0,) * 4) #(0,) * 4这里0表示黑色填充，255表示白色填充
            # 使用alpha层的rot作为掩码创建一个复合图像
            out = Image.composite(im_rotate, fff, im_rotate)

            filename1 = os.path.splitext(filename)
            # print(filename1[0])
            filename2 = os.path.basename(filename1[0])
            # print(filename2)
            filename_out = rootdir_rotate + filename2 + "_" + str(angle_num + 1) + ".png"
            # print(filename_out)
            out.convert(im.mode).save(filename_out)

        ratio_fz=ratio_fz+len(angle_range)
        ratio_fm = (len(list)-1)  * len(angle_range)   # 计算总比例的分母
        ratio = float(ratio_fz) / ratio_fm*100
        print("总共%d个字体，当前已处理第%d个字体，处理进度：%0.1f%%" %(len(list)-1,num, ratio))
print("处理结束，大功告成！" )
