'''
code by haezu
'''

#-*- coding:utf-8 -*-
import cv2
import os

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
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    # 일반 Blur
    dst = cv2.blur(img, (7, 7))

    cv2.imwrite('C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/Blur/' + 'B_' + file_name, dst)
    count+=1