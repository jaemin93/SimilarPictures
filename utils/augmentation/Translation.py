#-*- coding:utf-8 -*-
import cv2
import numpy as np
import os

def im_trim (img, num): #함수로 만든다
    if num==2:
        x = 0; y = 0; #자르고 싶은 지점의 x좌표와 y좌표 지정
        w = 250; h = 300; #x로부터 width, y로부터 height를 지정
    elif num==1:
        x = 50; y = 0; #자르고 싶은 지점의 x좌표와 y좌표 지정
        w = 250; h = 300; #x로부터 width, y로부터 height를 지정
    img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
    return img_trim #필요에 따라 결과물을 리턴

count = 0
#이미지 파일들이 있는 경로
IMAGE_DIR_BASE = 'C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/train'
image_file_list = os.listdir(IMAGE_DIR_BASE)
print(image_file_list)
for file_name in image_file_list:
    print(count)
    count = count + 1
    image = cv2.imread(IMAGE_DIR_BASE + '/' + file_name)
    rows, cols = image.shape[:2]

    # X축으로 50 이동
    M1 = np.float32([[1,0,50],[0,1,0]])
    # X축으로 -5 이동
    M2 = np.float32([[1, 0, -50], [0, 1, 0]])
    dst1 = cv2.warpAffine(image, M1,(300, 300))
    dst2 = cv2.warpAffine(image, M2, (300, 300))
    dst1 = im_trim(dst1, 1)
    dst2 = im_trim(dst2, 2)

    dst1 = cv2.resize(dst1, (300, 300))
    dst2 = cv2.resize(dst2, (300, 300))

    cv2.imwrite('C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/Translation_R/' + 'T_' + file_name, dst1)
    cv2.imwrite('C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/Translation_L/' + 'T_' + file_name, dst2)


