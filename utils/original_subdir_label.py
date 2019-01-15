'''
tfrecord로 변환하기 위해 tf-slim 오픈소스를 사용하는데 이때
directory형태는 
label -- images . . . 
label -- images . . .
 .
 .
 .           
 되어있기 때문에 기존 directory의 구조를 바꿔준다.           
'''

import os
import cv2

def main():
    ORIGINAL_IMAGE = 'C:\\Users\\iceba\\develop\\data\\dummy\\img\\naver_photos\\original'
    DIVIDE_IMAGE = 'C:\\Users\\iceba\\develop\\data\\dummy\\img\\naver_photos\\total'

    label_check_list = list()
    idx = 1

    for img in os.listdir(ORIGINAL_IMAGE):
        original_image = ORIGINAL_IMAGE + os.sep + img
        label = img.split('_')[0]
        if not label in label_check_list:
            print(str(idx)+' directory success')
            label_check_list.append(str(label))
            os.mkdir(DIVIDE_IMAGE+os.sep+str(idx))
            idx += 1
        # print(DIVIDE_IMAGE+os.sep+str(idx-1)+os.sep+img)
        copy_image = cv2.imread(original_image)
        cv2.imwrite(DIVIDE_IMAGE+os.sep+str(idx-1)+os.sep+img, copy_image)

if if __name__ == "__main__":
    _main()
