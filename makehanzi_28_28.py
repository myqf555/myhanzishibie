#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'david'
# 实现某一汉字多种字体图片打印输出

# import  codecs
import os
import pygame
import sys
import importlib
importlib.reload(sys)

#上面3行代码是为了防止发生“UnicodeEncodeError: ‘ascii’ codec can’t encode”错误

chinese_dir = 'chinese'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)

pygame.init()

dic = {'人':'ren','不':'bu','风':'feng','无':'wu','花':'hua','春':'chun','一':"yi",'山':'shan',\
       '月':'yue','时':'shi','時':'shi','云':'yun','雲':'yun','来':'lai','來':'lai','日':'ri', '天':'tian',\
       '夜': 'ye', '有': 'you', '江': 'jiang', '水': 'shui', '长': 'chang', '長': 'chang', '年': 'nian'}
# word = unichr(start)
# word1=str(word) #utf8转化为ascii
# ops = dic.get(word1)
word='时'
ops = dic.get(word)
print(word,ops)

rootdir = "F://pythonwork/shengchenghanzi/font/"
list = os.listdir(rootdir) #列出文件夹下所有的文件
for num in range(0,len(list)):
        path = os.path.join(rootdir,list[num])
        if os.path.isfile(path):
            filename= os.path.abspath(path)  # 返回文件名
            file=os.path.basename(path)
            print(filename)
            print(file)
            font = pygame.font.Font(filename, 28)
            rtext = font.render(word, True, (255, 255, 255),(0,0,0))
            pygame.image.save(rtext, os.path.join(chinese_dir, ops+str(num+1) + ".png"))#用于输出标准格式文件
            # pygame.image.save(rtext, os.path.join(chinese_dir, file + ".png"))#用于测试新增输入法能否正常显示
            #os.system("pause")

