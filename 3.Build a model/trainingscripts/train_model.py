#!/usr/bin/env python
# coding: utf-8
# Extended from: https://github.com/mlmed/torchxrayvision/blob/master/scripts/train_model.py
import os,sys
import os,sys,inspect
from glob import glob
from os.path import exists, join
#import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torchvision, torchvision.transforms
import sklearn, sklearn.model_selection

import random
import train_utils
import torchxrayvision as xrv

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from padchest_dataset import PC_Dataset_Custom
import multiprocessing

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default="", help='')
    parser.add_argument('-name', type=str, default='densenet')
    parser.add_argument('--output_dir', type=str, default=".\outputs")
    parser.add_argument('--dataset', type=str, default="pc")
    parser.add_argument('--dataset_dir', type=str, default="F:\publicimagingdatasets\padchest")
    parser.add_argument('--custom_padchest_csv_file', type=str, default=r"PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
    parser.add_argument('--model', type=str, default="densenet")
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--cuda', type=bool, default=True, help='')
    parser.add_argument('--num_epochs', type=int, default=20, help='')
    parser.add_argument('--batch_size_per_gpu', type=int, default=24, help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(), help='')
    parser.add_argument('--taskweights', type=bool, default=True, help='')
    parser.add_argument('--featurereg', type=bool, default=False, help='')
    parser.add_argument('--weightreg', type=bool, default=False, help='')
    parser.add_argument('--data_aug', type=bool, default=True, help='')
    parser.add_argument('--data_aug_rot', type=int, default=45, help='')
    parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
    parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
    parser.add_argument('--label_concat', type=bool, default=False, help='')
    parser.add_argument('--label_concat_reg', type=bool, default=False, help='')
    parser.add_argument('--multicuda', type=bool, default=True, help='')
    parser.add_argument('--azure_ml', type=bool, default=False, help='')
    parser.add_argument('--testing', type=bool, default=False, help='')
    
    cfg = parser.parse_args()
    print(cfg)

    #Sets output directory as required by AzureML studio
    if cfg.azure_ml:
        cfg.output_dir = './outputs'

    if cfg.testing:
        cfg.limit = 10
    else:
        cfg.limit = None

    if torch.cuda.device_count() > 1:
        cfg.batch_size_per_gpu =  torch.cuda.device_count() * cfg.batch_size_per_gpu

    data_aug = None
    if cfg.data_aug:
        data_aug = torchvision.transforms.Compose([
            xrv.datasets.ToPILImage(),
            torchvision.transforms.RandomAffine(cfg.data_aug_rot,
                                                translate=(cfg.data_aug_trans, cfg.data_aug_trans),
                                                scale=(1.0-cfg.data_aug_scale, 1.0+cfg.data_aug_scale)),
            torchvision.transforms.ToTensor()
        ])
        print(data_aug)

    # Set resolution according to Densenet
    transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

    cfg.custom_padchest_csv_file = cfg.dataset_dir + os.sep + cfg.custom_padchest_csv_file
    cfg.dataset_dir = cfg.dataset_dir + os.sep + 'png'
    datas = []
    datas_names = []

    # Customized dataset
    if "pc" in cfg.dataset:
        #Custom dataset
        dataset = PC_Dataset_Custom(
            imgpath=cfg.dataset_dir, 
            csvpath=cfg.custom_padchest_csv_file,
            transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA"], flat_dir=False)
        datas.append(dataset)
        datas_names.append("pc")
    print("datas_names", datas_names)

    dataset = xrv.datasets.Merge_Dataset(datas)

    # Setting the seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.cuda:
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # You can select different network architectures to train
    if "densenet" in cfg.model:
        model = xrv.models.DenseNet(num_classes=dataset.labels.shape[1], in_channels=1,
                                    **xrv.models.get_densenet_params(cfg.model))
    elif "resnet101" in cfg.model:
        model = torchvision.models.resnet101(num_classes=dataset.labels.shape[1], pretrained=False)
        #patch for single channel
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    elif "resnet50" in cfg.model:
        model = torchvision.models.resnet50(num_classes=dataset.labels.shape[1], pretrained=False)
        #patch for single channel
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    elif "shufflenet_v2_x2_0" in cfg.model:
        model = torchvision.models.shufflenet_v2_x2_0(num_classes=dataset.labels.shape[1], pretrained=False)
        #patch for single channel
        model.conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
    elif "squeezenet1_1" in cfg.model:
        model = torchvision.models.squeezenet1_1(num_classes=dataset.labels.shape[1], pretrained=False)
        #patch for single channel
        model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
    else:
        raise Exception("no model")


    train_utils.train(model, dataset, cfg)
