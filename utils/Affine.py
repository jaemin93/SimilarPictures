'''
code by haezu
'''

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def load_image(addr):
    img = open(addr)
    return img

count = 0
#이미지 파일들이 있는 경로
IMAGE_DIR_BASE = 'C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/train'
image_file_list = os.listdir(IMAGE_DIR_BASE)
for file_name in image_file_list:
    print(count)
    img = cv2.imread(IMAGE_DIR_BASE + '/' + file_name)
    rows, cols, ch = img.shape

    img = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)

    pts1 = np.float32([[0, 0], [300, 100], [100, 100]])
    pts2 = np.float32([[30, 60], [360, 60], [130, 160]])

    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imwrite('C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/Affine/' + 'A_' + file_name, dst)
    count+=1