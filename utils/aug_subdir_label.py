'''
naver에서 받은 데이터셋과 augmentation한 데이터 폴더안에 이미지들을 
모델별로 분류하여 모델이름이 곧 label이 되어서 sub directory가 되고 
분류된 데이터셋을 만듭니다. 

분류된 폴더 트리구조는 상위 디렉토리에서 data_folder_view.txt에서 확인가능 합니다.
'''
import os
import cv2
from operator import eq

def make_label_dict(path, d):
    total_path = os.listdir(path)
    for idx, sub_label in enumerate(total_path):
        label_dir = os.listdir(path + os.sep + sub_label)
        label = 'none'
        for img_name in label_dir:
            label = img_name.split('_')[0]
        d[label] = sub_label


# 트레이닝 데이터셋
AUG_IMG_DIR = 'C:\\Users\\iceba\\develop\\data\\dummy\\img\\naver_photos\\augmentation'
TOTAL_IMG_DIR = 'C:\\Users\\iceba\\develop\\data\\dummy\\img\\naver_photos\\total'
count = 0
idx = 1
d = dict()
make_label_dict(TOTAL_IMG_DIR, d)

# tfrecord 형태로 바꾸기위한 디렉토리 변환
for sub_idx, dir_list in enumerate(os.listdir(AUG_IMG_DIR)):
    sub_dir_list = AUG_IMG_DIR + os.sep + str(dir_list)
    for data_idx, img_name in enumerate(os.listdir(sub_dir_list)):
        #print(data_idx, img_name) 진행상황을 보기 위함
        img_info = img_name.split('_')
        model_id = img_info[1]
        product_id = img_info[2]
        aug_info = img_info[0]
        img_id = img_info[3][:img_info[3].find('.')]

        #Trans Augmentation이 L, R로 나누어져있어서 두가지로 나눈다.
        if eq(aug_info, 'T') and sub_idx == 3:
            aug_info = 'L'
        elif eq(aug_info, 'T') and sub_idx == 4:
            aug_info = 'R'
        file_name = model_id + '_' + product_id + '_' + img_id + '_' + aug_info + '.jpg'
        img = cv2.imread(sub_dir_list + os.sep + img_name)
        # print('C:\\Users\\iceba\\develop\\data\\naver\\naver_photos'+ os.sep + \
        #             model_id + os.sep + file_name)
        print('C:\\Users\\iceba\\develop\\data\\dummy\\img\\naver_photos\\total'+ os.sep + \
                    d[model_id] + os.sep + file_name)
        cv2.imwrite('C:\\Users\\iceba\\develop\\data\\dummy\\img\\naver_photos\\total'+ os.sep + \
                    d[model_id] + os.sep + file_name, img)