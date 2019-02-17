import numpy as np
import cv2
from matplotlib import pyplot as plt
import pylab
import os

IMAGE_DIR = 'C:\\Users\\iceba\\develop\\python\\naver_d2_fest_6th\\SimilarPictures\\img\\feature_mapping2'
SAVE_DIR = 'C:\\Users\\iceba\\develop\\python\\naver_d2_fest_6th\\SimilarPictures\\img\\background'

for img_dir in os.listdir(IMAGE_DIR):
    img_file = IMAGE_DIR + os.sep + img_dir
    save_img = SAVE_DIR + os.sep + img_dir
    img = cv2.imread(img_file)
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    cv2.imwrite(save_img, img)