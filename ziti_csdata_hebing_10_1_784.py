#!/usr/bin/python3.6
# -*- coding:utf-8 -*-
__author__ = 'david'
import numpy as np

CS=np.load('ziti_csdata_10_1_784.npz')
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

cs_image_temp=np.vstack((image_ren1_cs,image_bu2_cs,image_feng3_cs,image_wu4_cs,image_hua5_cs, \
                        image_chun6_cs,image_yi7_cs,image_shan8_cs,image_tian9_cs,image_yue10_cs))
cs_label_temp=np.vstack((label_ren1_cs,label_bu2_cs,label_feng3_cs,label_wu4_cs,label_hua5_cs, \
                        label_chun6_cs,label_yi7_cs,label_shan8_cs,label_tian9_cs,label_yue10_cs))

data_cs = np.hstack([cs_image_temp, cs_label_temp])

np.savez('ziti_csdata_hebing_10_1_784.npz',data_cs=data_cs)



