#!/usr/bin/python3.6
# -*- coding:utf-8 -*-
__author__ = 'david'
import numpy as np

XL=np.load('ziti_xldata_10_1_784.npz')
image_ren1_xl=XL['image_ren1_xl']
label_ren1_xl=XL['label_ren1_xl']
image_bu2_xl=XL['image_bu2_xl']
label_bu2_xl=XL['label_bu2_xl']
image_feng3_xl=XL['image_feng3_xl']
label_feng3_xl=XL['label_feng3_xl']
image_wu4_xl=XL['image_wu4_xl']
label_wu4_xl=XL['label_wu4_xl']
image_hua5_xl=XL['image_hua5_xl']
label_hua5_xl=XL['label_hua5_xl']
image_chun6_xl=XL['image_chun6_xl']
label_chun6_xl=XL['label_chun6_xl']
image_yi7_xl=XL['image_yi7_xl']
label_yi7_xl=XL['label_yi7_xl']
image_shan8_xl=XL['image_shan8_xl']
label_shan8_xl=XL['label_shan8_xl']
image_tian9_xl=XL['image_tian9_xl']
label_tian9_xl=XL['label_tian9_xl']
image_yue10_xl=XL['image_yue10_xl']
label_yue10_xl=XL['label_yue10_xl']


xl_image_temp=np.vstack((image_ren1_xl,image_bu2_xl,image_feng3_xl,image_wu4_xl,image_hua5_xl, \
                        image_chun6_xl,image_yi7_xl,image_shan8_xl,image_tian9_xl,image_yue10_xl))
xl_label_temp=np.vstack((label_ren1_xl,label_bu2_xl,label_feng3_xl,label_wu4_xl,label_hua5_xl, \
                        label_chun6_xl,label_yi7_xl,label_shan8_xl,label_tian9_xl,label_yue10_xl))

data_xl = np.hstack([xl_image_temp, xl_label_temp])

np.savez('ziti_xldata_hebing_10_1_784.npz',data_xl=data_xl)



