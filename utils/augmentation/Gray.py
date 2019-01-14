import cv2
import os
count = 0
#이미지 파일들이 있는 경로
IMAGE_DIR_BASE = 'C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/train'
image_file_list = os.listdir(IMAGE_DIR_BASE)
for file_name in image_file_list:
    image = cv2.imread(IMAGE_DIR_BASE + '/' + file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # h, s, v = cv2.split(hsv)
    # v = v - 100
    # # print(v)
    # image = cv2.merge([h, s, v])
    # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    cv2.imwrite('C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/Gray/' + 'G_' + file_name, image)
    count = count + 1
    print(count)