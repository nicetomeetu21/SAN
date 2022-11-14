# -*- coding:utf-8 -*-
import cv2 as cv
import os
import torch
from natsort import natsorted
from torchvision import transforms
import numpy as np
def read_cube_to_np(img_dir, cvflag =  cv.IMREAD_GRAYSCALE):
    assert os.path.exists(img_dir), f"got {img_dir}"
    print(img_dir)
    imgs = []
    names = natsorted(os.listdir(img_dir))
    for name in names:
        img = cv.imread(os.path.join(img_dir, name),cvflag)
        imgs.append(img)
    imgs = np.stack(imgs, axis=2)
    return imgs


def read_cube_to_tensor(path, cvflag =  cv.IMREAD_GRAYSCALE):
    imgs = []
    names = natsorted(os.listdir(path))
    for name in names:
        img = cv.imread(os.path.join(path, name), cvflag)
        img = transforms.ToTensor()(img)
        imgs.append(img)
    imgs = torch.stack(imgs, dim=1)
    return imgs
