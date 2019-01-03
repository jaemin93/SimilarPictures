import os
import cv2
import random

test_img_path = "C:\\Users\\iceba\\develop\\data\\naver_train"
val = 'C:\\Users\\iceba\\develop\\data\\naver_result\\vali'

test_idx = random.sample(range(0, 10617), 200)
l = list()
for idx, img in enumerate(os.listdir(test_img_path)):
    l.append([idx, img])
    if idx in test_idx:
        pass
        # print(idx)
        # val_img = cv2.imread(test_img_path + os.sep + img)
        # cv2.imwrite(val+os.sep+img,val_img)

for i in test_idx:
    cpy_path = test_img_path + os.sep + str(l[i][1])
    val_img = cv2.imread(cpy_path)
    cv2.imwrite(val+os.sep+str(l[i][1]), val_img)
