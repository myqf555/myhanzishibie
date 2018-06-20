#!/usr/bin/python3.6
# -*- coding:utf-8 -*-
__author__ = 'david'
import numpy as np

CS=np.load('ziti_csdata_20_1_784.npz')
image_ren1_cs=CS['image_ren1_cs']
label_ren1_cs=CS['label_ren1_cs']
image_bu2_cs=CS['image_bu2_cs']
label_bu2_cs=CS['label_bu2_cs']
image_feng3_cs=CS['image_feng3_cs']
label_feng3_cs=CS['label_feng3_cs']
image_wu4_cs=CS['image_wu4_cs']
label_wu4_cs=CS['label_wu4_cs']
image_hua5_cs=CS['image_hua5_cs']
label_hua5_cs=CS['label_hua5_cs']
image_chun6_cs=CS['image_chun6_cs']
label_chun6_cs=CS['label_chun6_cs']
image_yi7_cs=CS['image_yi7_cs']
label_yi7_cs=CS['label_yi7_cs']
image_shan8_cs=CS['image_shan8_cs']
label_shan8_cs=CS['label_shan8_cs']
image_tian9_cs=CS['image_tian9_cs']
label_tian9_cs=CS['label_tian9_cs']
image_yue10_cs=CS['image_yue10_cs']
label_yue10_cs=CS['label_yue10_cs']
image_shi11_cs = CS['image_shi11_cs']
label_shi11_cs = CS['label_shi11_cs']
image_yun12_cs = CS['image_yun12_cs']
label_yun12_cs = CS['label_yun12_cs']
image_lai13_cs = CS['image_lai13_cs']
label_lai13_cs = CS['label_lai13_cs']
image_ri14_cs = CS['image_ri14_cs']
label_ri14_cs = CS['label_ri14_cs']
image_ye15_cs = CS['image_ye15_cs']
label_ye15_cs = CS['label_ye15_cs']
image_you16_cs = CS['image_you16_cs']
label_you16_cs = CS['label_you16_cs']
image_jiang17_cs = CS['image_jiang17_cs']
label_jiang17_cs = CS['label_jiang17_cs']
image_shui18_cs = CS['image_shui18_cs']
label_shui18_cs = CS['label_shui18_cs']
image_chang19_cs = CS['image_chang19_cs']
label_chang19_cs = CS['label_chang19_cs']
image_nian20_cs = CS['image_nian20_cs']
label_nian20_cs = CS['label_nian20_cs']

cs_image_temp = np.vstack((image_ren1_cs, image_bu2_cs, image_feng3_cs, image_wu4_cs, image_hua5_cs,\
                           image_chun6_cs, image_yi7_cs, image_shan8_cs, image_tian9_cs, image_yue10_cs,\
                           image_shi11_cs, image_yun12_cs, image_lai13_cs, image_ri14_cs, image_ye15_cs,\
                           image_you16_cs, image_jiang17_cs, image_shui18_cs, image_chang19_cs, image_nian20_cs))
cs_label_temp = np.vstack((label_ren1_cs, label_bu2_cs, label_feng3_cs, label_wu4_cs, label_hua5_cs,\
                           label_chun6_cs, label_yi7_cs, label_shan8_cs, label_tian9_cs, label_yue10_cs,\
                           label_shi11_cs, label_yun12_cs, label_lai13_cs, label_ri14_cs, label_ye15_cs,\
                           label_you16_cs, label_jiang17_cs, label_shui18_cs, label_chang19_cs, label_nian20_cs))

data_cs = np.hstack([cs_image_temp, cs_label_temp])

np.savez('ziti_csdata_hebing_20_1_784.npz',data_cs=data_cs)



