# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   trainer.py
@Time    :   2022/3/1 20:06
@Desc    :
"""
import os
import cv2
import h5py
import copy
import utils
import torch
import numpy as np
from loss import Loss
from data import augment
from options import args
from data import get_loader
from models import get_model
from scheduler import create_scheduler
from adp_utils import train, evaluation
from utils import make_optimizer, imresize

for path in ['/data/zhwzhong', '/root/autodl-fs', '/home/wcy']:
    tmp_path = '{}/Data'.format(path)
    if os.path.exists(tmp_path):
        root_path = tmp_path


def random_augment(img):
    pass


def evaluate(model, val_data, device, criterion):
    sv_path = f'./zssr/{args.dataset}/x{args.scale}/'
    for _, samples in val_data:
        samples = utils.to_device(samples, device)
        out = utils.ensemble(samples, model, args.ensemble_mode, args.dataset) if args.self_ensemble else model(samples)
        torch.cuda.synchronize()
        loss = criterion(out['img_out'], samples['img_gt'])

        if args.save_result:
            for index in range(samples['img_gt'].size(0)):
                save_name = os.path.join(sv_path, samples['img_name'][0])
                img = utils.tensor2uint(out['img_out'][index: index + 1], data_range=args.data_range)
                cv2.imwrite(save_name, img)
                # print('Image Saved to {}'.format(save_name))
        metrics = utils.calc_metrics(out['img_out'], samples['img_gt'], args)


def zero_sisr():
    sum_psnr = []
    model = get_model(args)

    criterion = Loss(args)
    device = torch.device(args.device)

    model.to(device)
    optimizer = make_optimizer(args, model)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(args.num_gpus)))
    model_without_ddp = model.module

    model_path = args.load_name

    try:
        checkpoint = torch.load(args.load_name, map_location='cuda:{}'.format(args.local_rank))
        model_without_ddp.load_state_dict(checkpoint['model'])
    except FileNotFoundError:
        print('===> File {} not exists'.format(model_path))
    else:
        print('===> File {} loaded'.format(model_path))

    val_list = ['015_02_D1_th.bmp', '247_01_D4_th.bmp', '003_02_D1_th.bmp', '027_02_D1_th.bmp', '025_01_D4_th.bmp',
                '031_01_D4_th.bmp', '030_01_D3_th.bmp', '032_01_D1_th.bmp', '033_01_D2_th.bmp', '148_01_D4_th.bmp',
                '163_01_D3_th.bmp', '170_01_D3_th.bmp', '072_01_D1_th.bmp', '484_01_D3_th.bmp', '018_01_D2_th.bmp',
                '025_01_D3_th.bmp', '093_02_D1_th.bmp', '082_01_D3_th.bmp', '504_01_D3_th.bmp', '136_01_D3_th.bmp']

    if args.test_name == 'val':
        train_data = h5py.File(f'{root_path}/PBVS2024/h5py/sisr_train.h5', 'r')
        train_data = [np.array(train_data['GT'][key]) for key in val_list]
    else:
        train_data = h5py.File(f'{root_path}/PBVS2024/h5py/sisr_test.h5', 'r')
        train_data = [np.array(train_data['GT'][key]) for key in train_data['GT'].keys()]

    for gt_img in train_data:
        lr_img = cv2.resize(gt_img, fx=1 / 8, fy=1 / 8, dsize=None)
        lr_img = np.expand_dims(imresize(lr_img.astype(float), scalar_scale=8), 0)

        samples = {
            'lr_up': torch.from_numpy(lr_img / 255).unsqueeze(0).float(),
            'img_gt': torch.from_numpy(gt_img / 255).unsqueeze(0).unsqueeze(0).float(),
        }
        out = evaluation(model, samples, device, args)
        metrics1 = utils.calc_metrics(out['img_out'], samples['img_gt'], args)['PSNR'].item()
        # print('===>Before Adaption, PSNR : {:.<3f}'.format(metrics1))

        psnr_gain = 0
        for epoch in range(args.epochs):
            lr_tensor = []
            gt_tensor = []
            for _ in range(args.batch_size):
                tmp_gt = cv2.resize(gt_img, fx=1 / 8, fy=1 / 8, dsize=None)
                # tmp_gt = imresize(tmp_gt.astype(float), scalar_scale=args.scale).astype(np.uint8)
                tmp_gt = augment.get_patch(np.expand_dims(tmp_gt, 0), patch_size=args.patch_size, scale=1)
                tmp_gt = augment.random_rot(tmp_gt, hflip=True, rot=True)

                tmp_lr = cv2.resize(tmp_gt.squeeze(), fx=1 / 8, fy=1 / 8, dsize=None)
                tmp_lr = np.expand_dims(imresize(tmp_lr.astype(float), scalar_scale=8), 0)

                tmp_lr, tmp_gt = augment.np_to_tensor(tmp_lr, tmp_gt, input_data_range=255)

                lr_tensor.append(tmp_lr.reshape(1, 1, args.patch_size, args.patch_size))
                gt_tensor.append(tmp_gt.reshape(1, 1, args.patch_size, args.patch_size))

            sample = {'lr_up': torch.cat(lr_tensor, dim=0), 'img_gt': torch.cat(gt_tensor, dim=0)}
            train_loss = train(model, criterion, sample, optimizer, device, args)

            lr_scheduler.step(epoch)

            samples = {
                'lr_up': torch.from_numpy(lr_img / 255).unsqueeze(0).float(),
                'img_gt': torch.from_numpy(gt_img / 255).unsqueeze(0).unsqueeze(0).float(),
            }
            out = evaluation(model, samples, device, args)
            torch.cuda.empty_cache()
            metrics2 = utils.calc_metrics(out['img_out'], samples['img_gt'], args)['PSNR'].item()
            psnr_gain = metrics2 - metrics1
            # print(f'===>Epoch {epoch}, Loss {round(train_loss, 5)}, After Adaption, PSNR GAIN: {round(psnr_gain, 5)}')
        sum_psnr.append(psnr_gain)
        print(f'===>After Adaption, PSNR GAIN: {psnr_gain}')
    print('Total Gain', np.mean(sum_psnr))


def zero_gdsr():
    sum_psnr = []
    model = get_model(args)

    criterion = Loss(args)
    device = torch.device(args.device)

    model.to(device)
    optimizer = make_optimizer(args, model)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(args.num_gpus)))
    model_without_ddp = model.module

    model_path = args.load_name

    try:
        checkpoint = torch.load(args.load_name, map_location='cuda:{}'.format(args.local_rank))
        model_without_ddp.load_state_dict(checkpoint['model'])
    except FileNotFoundError:
        print('===> File {} not exists'.format(model_path))
    else:
        print('===> File {} loaded'.format(model_path))

    val_x8_list = ['380_01_D3_th.bmp', '020_01_D3_th.bmp', '027_01_D2_th.bmp', '059_01_D1_th.bmp',
                   '427_01_D3_th.bmp', '174_01_D3_th.bmp', '059_01_D4_th.bmp', '109_01_D1_th.bmp',
                   '044_01_D3_th.bmp', '063_01_D1_th.bmp', '074_02_D1_th.bmp', '117_02_D1_th.bmp',
                   '127_01_D4_th.bmp', '225_01_D4_th.bmp', '421_01_D3_th.bmp', '092_01_D4_th.bmp',
                   '050_01_D2_th.bmp', '047_01_D1_th.bmp', '035_01_D1_th.bmp', '008_02_D1_th.bmp',
                   '013_01_D2_th.bmp', '082_02_D1_th.bmp', '085_01_D1_th.bmp', '073_02_D1_th.bmp',
                   '074_02_D1_th.bmp', '074_01_D2_th.bmp', '074_02_D1_th.bmp', '094_01_D1_th.bmp',
                   '098_02_D1_th.bmp', '163_01_D3_th.bmp', '123_01_D1_th.bmp', '113_02_D1_th.bmp',
                   '111_01_D1_th.bmp', '107_01_D1_th.bmp', '102_01_D1_th.bmp', '095_01_D2_th.bmp',
                   '095_01_D1_th.bmp', '091_01_D2_th.bmp', '068_01_D4_th.bmp', '062_01_D3_th.bmp']

    val_x16_list = ['042_01_D1_th.bmp', '169_01_D4_th.bmp', '319_01_D3_th.bmp', '332_01_D4_th.bmp',
                    '477_01_D4_th.bmp', '139_01_D4_th.bmp', '077_01_D4_th.bmp', '044_01_D4_th.bmp',
                    '023_02_D1_th.bmp', '242_01_D4_th.bmp', '260_01_D4_th.bmp', '279_01_D4_th.bmp',
                    '281_01_D3_th.bmp', '306_01_D3_th.bmp', '310_01_D3_th.bmp', '345_01_D4_th.bmp',
                    '353_01_D3_th.bmp', '356_01_D3_th.bmp', '520_01_D3_th.bmp', '612_01_D4_th.bmp',
                    '683_01_D4_th.bmp', '735_01_D4_th.bmp', '743_01_D4_th.bmp', '752_01_D4_th.bmp',
                    '637_01_D4_th.bmp', '668_01_D4_th.bmp', '493_01_D3_th.bmp', '467_01_D4_th.bmp',
                    '474_01_D4_th.bmp', '470_01_D3_th.bmp', '277_01_D3_th.bmp', '235_01_D3_th.bmp',
                    '091_01_D1_th.bmp', '078_01_D4_th.bmp', '233_01_D4_th.bmp', '249_01_D4_th.bmp',
                    '279_01_D4_th.bmp', '330_01_D4_th.bmp', '363_01_D4_th.bmp', '409_01_D4_th.bmp',
                    ]

    if args.test_name == 'val':
        train_data = h5py.File(f'{root_path}/PBVS2024/h5py/gdsr_train.h5', 'r')
        train_gt = [np.array(train_data['GT'][key]) for key in f'val_x{args.scale}_list']
        train_rgb = [np.array(train_data['RGB'][key]) for key in f'val_x{args.scale}_list']
    else:
        train_data = h5py.File(f'{root_path}/PBVS2024/h5py/gdsr_test.h5', 'r')
        train_gt = [np.array(train_data['GT'][key]) for key in train_data['GT'].keys()]
        train_rgb = [np.array(train_data['RGB'][key]) for key in train_data['GT'].keys()]

    for index in range(len(train_gt)):
        gt_img, rgb_img = train_gt[index], np.transpose(train_rgb[index], (2, 0, 1))
        lr_img = cv2.resize(gt_img, fx=1 / args.scale, fy=1 / args.scale, dsize=None)
        lr_img = np.expand_dims(imresize(lr_img.astype(float), scalar_scale=args.scale), 0)

        samples = {
            'img_rgb': torch.from_numpy(rgb_img / 255).float(),
            'lr_up': torch.from_numpy(lr_img / 255).unsqueeze(0).float(),
            'img_gt': torch.from_numpy(gt_img / 255).unsqueeze(0).unsqueeze(0).float(),
        }
        out = evaluation(model, samples, device, args)
        metrics1 = utils.calc_metrics(out['img_out'], samples['img_gt'], args)['PSNR'].item()

        print('===>Before Adaption, PSNR : {:.<3f}'.format(metrics1), end=', ')

        lr_tensor = []
        gt_tensor = []
        rgb_tensor = []

        for epoch in range(num_epochs):
            for _ in range(args.batch_size):
                tmp_gt = cv2.resize(gt_img, fx=1 / args.scale, fy=1 / args.scale, dsize=None)
                tmp_rgb = cv2.resize(rgb_img, fx=1 / args.scale, fy=1 / args.scale, dsize=None)

                tmp_gt, tmp_rgb = augment.get_patch(np.expand_dims(tmp_gt, 0), tmp_rgb, patch_size=args.patch_size, scale=1)
                tmp_gt, tmp_rgb = augment.random_rot(tmp_gt, tmp_rgb, hflip=True, rot=True)

                tmp_lr = cv2.resize(tmp_gt.squeeze(), fx=1 / args.scale, fy=1 / args.scale, dsize=None)
                tmp_lr = np.expand_dims(imresize(tmp_lr.astype(float), scalar_scale=8), 0)

                tmp_lr, tmp_gt, tmp_rgb = augment.np_to_tensor(tmp_lr, tmp_gt, tmp_rgb, input_data_range=255)

                lr_tensor.append(tmp_lr.reshape(1, 1, args.patch_size, args.patch_size))
                gt_tensor.append(tmp_gt.reshape(1, 1, args.patch_size, args.patch_size))
                rgb_tensor.append(tmp_rgb.reshape(1, 3, args.patch_size, args.patch_size))

            sample = {
                'lr_up': torch.cat(lr_tensor, dim=0),
                'img_gt': torch.cat(gt_tensor, dim=0),
                'img_rgb': torch.cat(rgb_tensor, dim=0)
            }
            train_loss = train(model, criterion, sample, optimizer, device)
            lr_scheduler.step(epoch)
            with torch.no_grad():
                out = evaluation(model, samples, device, args)
            metrics2 = utils.calc_metrics(out['img_out'], samples['img_gt'], args)['PSNR'].item()
            print('===>Loss {}, After Adaption, PSNR : {:.<3f}'.format(train_loss, metrics2), end=', ')
        sum_psnr.append(metrics2 - metrics1)
    print(np.mean(sum_psnr))


if __name__ == '__main__':
    zero_sisr()
    # zero_gdsr()