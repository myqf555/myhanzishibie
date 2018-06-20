#!/usr/bin/python3.6
# -*- coding:utf-8 -*-
__author__ = 'david'

from PIL import Image
import pytesseract

#设置为简体中文文字的识别
text=pytesseract.image_to_string(Image.open('C://Users/david/data/shouxie_hanzi/666.jpg'),lang='chi_sim')
#设置为繁体中文文字的识别
# text=pytesseract.image_to_string(Image.open('C://Users/david/data/shouxie_hanzi/111.png'),lang='chi_tra')

#设置为英文或阿拉伯字母的识别
# text=pytesseract.image_to_string(Image.open('C://Users/david/data/111.png'),lang='eng')
print(text)

