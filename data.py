# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   data.py
@Time    :   2021/12/20 09:04
@Desc    :
"""
import glob
import numpy as np
from PIL import Image

dataPath = {
    'train': '/data/zhwzhong/Data/DIV2K/DIV2K_train_HR/',
    'Set5': '/data/zhwzhong/Data/DIV2K/benchmark/Set5/HR/',
    'B100': '/data/zhwzhong/Data/DIV2K/benchmark/B100/HR/',
    'Set14': '/data/zhwzhong/Data/DIV2K/benchmark/Set14/HR/',
    'Urban100': '/data/zhwzhong/Data/DIV2K/benchmark/Urban100/HR/',
}

import tqdm
import time
from image_resize import imresize

time_read = 0
time_inter = 0
for i in range(2):
    for img_name in tqdm.tqdm(glob.glob(dataPath['train'] + '*.png')[: 5]):
        img_hr = np.array(Image.open(img_name)).astype(np.uint8)

        start = time.time()

        _ = imresize(img_hr, scalar_scale=2).astype(np.uint8)
        _ = imresize(img_hr, scalar_scale=3).astype(np.uint8)
        _ = imresize(img_hr, scalar_scale=4).astype(np.uint8)

        time_inter += (time.time() - start)

        start = time.time()

        a = np.array(Image.open(img_name.replace('HR', 'LR_bicubic/X2').replace('.png', 'x2.png'))).astype(np.uint8)
        b = np.array(Image.open(img_name.replace('HR', 'LR_bicubic/X3').replace('.png', 'x3.png'))).astype(np.uint8)
        c = np.array(Image.open(img_name.replace('HR', 'LR_bicubic/X4').replace('.png', 'x4.png'))).astype(np.uint8)

        time_read += (time.time() - start)

print(time_read, time_inter)