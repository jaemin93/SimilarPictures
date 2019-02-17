import os
import cv2
import random

def test_dataset(path, save_path):
    number = input('how many do you want number of test data?')
    for sub_dir in os.listdir(path):
        img_dir = path + os.sep + sub_dir
        test_data = random.sample(os.listdir(img_dir), int(len(img_dir)/int(number)))
        for img in test_data:
            print(img)
            cpy_img = cv2.imread(img_dir+os.sep+img)
            cv2.imwrite(save_path+os.sep+img, cpy_img)
    return True


if __name__ == "__main__":
    test_dataset('C:\\Users\\iceba\\develop\\data\\dummy\\img\\naver_photos\\total2', 'C:\\Users\\iceba\\develop\\python\\naver_d2_fest_6th\\SimilarPictures\\img\\test3')
    print('Done.')
