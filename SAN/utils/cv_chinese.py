# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


""" solve opencv-python not support chinese path """
def imread(filepath, flag=cv.IMREAD_GRAYSCALE):
    img = cv.imdecode(np.fromfile(filepath, dtype=np.uint8), flag)
    return img

def imwrite(filepath, img):
    cv.imencode('.png', img)[1].tofile(filepath)
