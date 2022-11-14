import time
import cv2 as cv
import numpy as np
import os
import torch
from natsort import natsorted
from torchvision import transforms

def solve_cubes(cube_root, cube_process):
    assert os.path.exists(cube_root)
    names = natsorted(os.listdir(cube_root))
    for name in names:
        st = time.time()
        cube_process(os.path.join(cube_root, name))
        print(name, time.time()-st)

def solve_cubes_with_result(cube_root, result_root, cube_process):
    assert os.path.exists(cube_root)
    names = natsorted(os.listdir(cube_root))
    for name in names:
        st = time.time()
        if not os.path.exists(os.path.join(result_root, name)): os.mkdir(os.path.join(result_root, name))
        cube_process(os.path.join(cube_root, name), os.path.join(result_root, name))
        print(name, time.time()-st)



def read_cube_to_np(img_dir, cvflag =  cv.IMREAD_GRAYSCALE):
    assert os.path.exists(img_dir), f"got {img_dir}"
    print(img_dir)
    imgs = []
    names = natsorted(os.listdir(img_dir))
    for name in names:
        img = cv.imread(os.path.join(img_dir, name),cvflag)
        imgs.append(img)
    imgs = np.stack(imgs, axis=0)
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

def save_cube(data, result_name, tonpy=False):
    if tonpy:
        np.save(result_name +'.npy', data)
    else:
        result_dir = result_name
        if not os.path.exists(result_dir): os.makedirs(result_dir)
        for i in range(data.shape[0]):
            cv.imwrite(os.path.join(result_dir, str(i+1)+'.png'), data[i, ...])