# -*- coding: utf-8 -*-
import cv2
import h5py
import numpy as np
from data import augment
from torch.utils import data
from utils.image_resize import imresize


def get_array(x, cached):
    return np.array(x) if cached else x

class PBVS(data.Dataset):
    def __init__(self, args, attr):
        self.args = args
        self.attr = attr
        self.file = h5py.File(f'./Data/sisr_{attr}.h5', 'r')

        self.img_names = [key for key in self.file['GT'].keys()]

        cached = (self.args.cached and attr == 'train')
        self.gt_imgs = [get_array(self.file['GT'].get(key), cached=cached) for key in self.img_names]
        self.lr_imgs = [get_array(self.file['LR'].get(key), cached=cached) for key in self.img_names]

    def __len__(self):
        return int(self.args.show_every * len(self.img_names)) if self.attr == 'train' else len(self.img_names)

    def __getitem__(self, item):
        item = item % len(self.gt_imgs)

        lr_img = imresize(np.array(self.lr_imgs[item]).astype(float), scalar_scale=self.args.scale)
        lr_img, gt_img = np.expand_dims(lr_img, 0), np.expand_dims(np.array(self.gt_imgs[item]), 0)

        if self.attr == 'train':
            lr_img, gt_img = augment.random_rot(augment.get_patch(lr_img, gt_img, patch_size=self.args.patch_size), hflip=True, rot=True)

        lr_img, gt_img = augment.np_to_tensor(lr_img / 255, gt_img / 255, input_data_range=1)

        return {'img_gt': gt_img, 'lr_up': lr_img, 'img_name': self.img_names[item]}

