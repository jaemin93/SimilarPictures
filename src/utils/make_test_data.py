import os
import cv2
import random

def test_dataset(path, test_path):
    number = input('how many do you want number of test data?')
    test_data = random.sample(os.listdir(path), int(number))
    for img in test_data:
        cpy_img = cv2.imread(path + os.sep + img)
        cv2.imwrite(test_path + os.sep + img, cpy_img)
    return True


if __name__ == "__main__":
    test_dataset('C:\\Users\\iceba\\develop\\python\\naver_d2_fest_6th\\img\\dummy', 'C:\\Users\\iceba\\develop\\python\\naver_d2_fest_6th\\img\\test')
    print('Done.')
