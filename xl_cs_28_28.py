#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'david'
# 按一定比例（3:7）将xxx_rotate文件夹下的图片分为训练样本和测试样本
import os
import shutil
mydir='nian20'
# yue10'/'tian9'/shan8'/'yi7 /'chun6'/'hua5'/'wu4'/'feng3'/'bu2'/'ren1'
#'nian20'/'chang19'/'shui18'/'jiang17'/'you16'/'ye15'/'ri14'/'lai13'/'yun12'/'shi11''
#=====================================================
os.chdir('F://pythonwork//shengchenghanzi//makehanzi_28_28/chinese//'+mydir+'//')
chinese_dir_xl = mydir+'_xl'
if not os.path.exists(chinese_dir_xl):
    os.mkdir(chinese_dir_xl)
chinese_dir_cs = mydir+'_cs'
if not os.path.exists(chinese_dir_cs):
    os.mkdir(chinese_dir_cs)
dir_cs="F://pythonwork//shengchenghanzi//makehanzi_28_28//chinese//"+mydir+"//"+chinese_dir_cs+"//"
dir_xl="F://pythonwork//shengchenghanzi//makehanzi_28_28//chinese//"+mydir+"//"+chinese_dir_xl+"//"
#=====================================================
rootdir = "F://pythonwork//shengchenghanzi//makehanzi_28_28//chinese//"+mydir+"//"+mydir+"_rotate//"

list = os.listdir(rootdir)  # 列出文件夹下所有的文件
ratio_fz = 0 #当前完成次数，计算总比例的分子
ratio_fm = len(list) #总数量
for num in range(0, len(list)):
    path = os.path.join(rootdir, list[num])  # 返回文件名
    # 所有信息30% 测试 70%训练
    if (num % 7) == 0:
        shutil.copyfile(path, dir_cs+list[num])

    else:
        shutil.copyfile(path, dir_xl+list[num])

    ratio = float(num+1) / ratio_fm*100
    print("总共有%d个图片待处理，当前已处理第%d个图片，处理进度：%0.2f%%" %(len(list),num+1, ratio))



