#!/usr/bin/python3.6
# -*- coding:utf-8 -*-
__author__ = 'david'
import numpy as np

XL = np.load('ziti_xldata_20_1_784.npz')
image_ren1_xl = XL['image_ren1_xl']
label_ren1_xl = XL['label_ren1_xl']
image_bu2_xl = XL['image_bu2_xl']
label_bu2_xl = XL['label_bu2_xl']
image_feng3_xl = XL['image_feng3_xl']
label_feng3_xl = XL['label_feng3_xl']
image_wu4_xl = XL['image_wu4_xl']
label_wu4_xl = XL['label_wu4_xl']
image_hua5_xl = XL['image_hua5_xl']
label_hua5_xl = XL['label_hua5_xl']
image_chun6_xl = XL['image_chun6_xl']
label_chun6_xl = XL['label_chun6_xl']
image_yi7_xl = XL['image_yi7_xl']
label_yi7_xl = XL['label_yi7_xl']
image_shan8_xl = XL['image_shan8_xl']
label_shan8_xl = XL['label_shan8_xl']
image_tian9_xl = XL['image_tian9_xl']
label_tian9_xl = XL['label_tian9_xl']
image_yue10_xl = XL['image_yue10_xl']
label_yue10_xl = XL['label_yue10_xl']
image_shi11_xl = XL['image_shi11_xl']
label_shi11_xl = XL['label_shi11_xl']
image_yun12_xl = XL['image_yun12_xl']
label_yun12_xl = XL['label_yun12_xl']
image_lai13_xl = XL['image_lai13_xl']
label_lai13_xl = XL['label_lai13_xl']
image_ri14_xl = XL['image_ri14_xl']
label_ri14_xl = XL['label_ri14_xl']
image_ye15_xl = XL['image_ye15_xl']
label_ye15_xl = XL['label_ye15_xl']
image_you16_xl = XL['image_you16_xl']
label_you16_xl = XL['label_you16_xl']
image_jiang17_xl = XL['image_jiang17_xl']
label_jiang17_xl = XL['label_jiang17_xl']
image_shui18_xl = XL['image_shui18_xl']
label_shui18_xl = XL['label_shui18_xl']
image_chang19_xl = XL['image_chang19_xl']
label_chang19_xl = XL['label_chang19_xl']
image_nian20_xl = XL['image_nian20_xl']
label_nian20_xl = XL['label_nian20_xl']

xl_image_temp = np.vstack((image_ren1_xl, image_bu2_xl, image_feng3_xl, image_wu4_xl, image_hua5_xl,\
                           image_chun6_xl, image_yi7_xl, image_shan8_xl, image_tian9_xl, image_yue10_xl,\
                           image_shi11_xl, image_yun12_xl, image_lai13_xl, image_ri14_xl, image_ye15_xl,\
                           image_you16_xl, image_jiang17_xl, image_shui18_xl, image_chang19_xl, image_nian20_xl))
xl_label_temp = np.vstack((label_ren1_xl, label_bu2_xl, label_feng3_xl, label_wu4_xl, label_hua5_xl,\
                           label_chun6_xl, label_yi7_xl, label_shan8_xl, label_tian9_xl, label_yue10_xl,\
                           label_shi11_xl, label_yun12_xl, label_lai13_xl, label_ri14_xl, label_ye15_xl,\
                           label_you16_xl, label_jiang17_xl, label_shui18_xl, label_chang19_xl, label_nian20_xl))

data_xl = np.hstack([xl_image_temp, xl_label_temp])
np.savez('ziti_xldata_hebing_20_1_784.npz', data_xl=data_xl)
