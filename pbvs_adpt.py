# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   pbvs_adpt.py
@Time    :   2023/2/17 11:44
@Desc    :
"""
import os
import cv2
import glob
import loss
import utils
import torch
import numpy as np
from options import args
from data import get_loader
from models import get_model
from trainer import evaluate
from data.nir import root_path
from utils import make_optimizer, down_sample
from scheduler import create_scheduler
from utils.image_resize import imresize
from adp_utils import train, evaluation
from data.augment import np_to_tensor, random_rot, get_patch

model = get_model(args)
device = torch.device(args.device)
model = torch.nn.parallel.DataParallel(model.to(device), device_ids=list(range(args.num_gpus)))

cp_path = f'./checkpoints/{args.dataset}/{args.file_name}'

if os.path.exists(args.load_name):
    cp_path = args.load_name
else:
    cp_path = f"{cp_path}/{args.load_name}"


if os.path.exists(cp_path):
    checkpoint = torch.load(cp_path)
    model.module.load_state_dict(checkpoint['model'])
else:
    print('Cannot load model ...')

lr_list = list(glob.glob(f'{root_path}/PBVS/track1/challengedataset/train/640_flir_hr/*.jpg'))
lr_list.extend(list(glob.glob(f'{root_path}/PBVS/track1/challengedataset/validation/640_flir_hr/*.jpg')))

val1_list = [11, 24, 25, 644, 959, 220, 316, 166, 139, 349, 400, 404, 638, 680, 683, 715, 752, 785, 1, 613]
val1_list = [str(img_name).zfill(4) + '.jpg' for img_name in val1_list]

val_list = sorted([
    img_name for img_name in lr_list if os.path.basename(img_name) in val1_list
])

test_list = glob.glob('/home/zhwzhong/Data/PBVS/track1/challengedataset/testingSetInput/evaluation1/hr_x4/*.jpg')

img_list = val_list if args.test_name == 'val' else test_list

args.lr = 1e-5

criterion = loss.Loss(args)
optimizer = make_optimizer(args, model)
lr_scheduler, num_epochs = create_scheduler(args, optimizer)
evaluate(model, criterion, args.test_name, device=device, val_data=get_loader(args, args.test_name), args=args)

for img_name in img_list:
    patch_size = args.patch_size // 8
    inp = cv2.imread(img_name, 0)
    gt_img = down_sample(inp, scale=1 / 4) if args.test_name == 'val' else inp
    gt_img = np.expand_dims(gt_img, 0)
    train_loss = []

    for epoch in range(200):
        lr_up = imresize(gt_img.squeeze().astype(float), 4)
        lr_up = torch.from_numpy(lr_up / 255).unsqueeze(0).unsqueeze(0).float()
        if epoch % 1 == 0:
            samples = {
                'lr_up': lr_up,
                'img_lr': torch.from_numpy(gt_img / 255).unsqueeze(0).float(),
                'img_gt': torch.from_numpy(inp / 255).unsqueeze(0).unsqueeze(0).float() if args.test_name == 'val' else lr_up
            }
            out = evaluation(model, samples, device, args)
            metrics = utils.calc_metrics(out['img_out'], samples['img_gt'], args)['PSNR'].item()
            print('===>Epochs: {:<3d}, Image: {}, PSNR : {:.<3f}'.format(epoch, os.path.basename(img_name), metrics), end=', ')

        lr_tensor, gt_tensor, up_tensor = [], [], []
        for _ in range(args.batch_size):
            tmp_gt = get_patch(gt_img, patch_size=patch_size)
            tmp_gt = random_rot(tmp_gt, hflip=True, rot=True)

            tmp_lr = down_sample(tmp_gt.squeeze(), 1 / 4)
            tmp_up = imresize(tmp_lr.astype(float), 4)
            tmp_lr, tmp_up, tmp_gt = np_to_tensor(tmp_lr, tmp_up, tmp_gt, input_data_range=255)
            up_tensor.append(tmp_up.reshape(1, 1, patch_size, patch_size))
            gt_tensor.append(tmp_gt.reshape(1, 1, patch_size, patch_size))
            lr_tensor.append(tmp_lr.reshape(1, 1, patch_size // 4, patch_size // 4))

        samples = {
            'img_gt': torch.cat(gt_tensor, dim=0),
            'lr_up': torch.cat(up_tensor, dim=0),
            'img_lr': torch.cat(lr_tensor, dim=0)
        }
        train_loss.append(train(model, criterion, samples, optimizer, device))
        if epoch % 1 == 0:
            print(np.mean(train_loss))






