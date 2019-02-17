import cv2
import numpy as np
import os

def Affine (input, file_name):
    img = cv2.resize(input, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    pts1 = np.float32([[0, 0], [300, 100], [100, 100]])
    pts2 = np.float32([[30, 60], [360, 60], [130, 160]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite('C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/Affine/' + 'A_' + file_name, dst)


def Blur (input, file_name):
    b, g, r = cv2.split(input)
    img = cv2.merge([r, g, b])
    # 일반 Blur
    dst = cv2.blur(img, (7, 7))
    cv2.imwrite('C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/Blur/' + 'B_' + file_name, dst)

def Flip (imput, file_name):
    img = cv2.flip(imput, 1)
    cv2.imwrite('C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/flip/' + 'F_' +file_name, img)

def im_trim (img, num): #함수로 만든다
    if num==2:
        x = 0; y = 0; #자르고 싶은 지점의 x좌표와 y좌표 지정
        w = 250; h = 300; #x로부터 width, y로부터 height를 지정
    elif num==1:
        x = 50; y = 0; #자르고 싶은 지점의 x좌표와 y좌표 지정
        w = 250; h = 300; #x로부터 width, y로부터 height를 지정
    img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
    return img_trim #필요에 따라 결과물을 리턴

def Translation (imput, file_name):
    # X축으로 50 이동
    M1 = np.float32([[1,0,50],[0,1,0]])
    # X축으로 -5 이동
    M2 = np.float32([[1, 0, -50], [0, 1, 0]])
    dst1 = cv2.warpAffine(imput, M1,(300, 300))
    dst2 = cv2.warpAffine(imput, M2, (300, 300))
    dst1 = im_trim(dst1, 1)
    dst2 = im_trim(dst2, 2)
    dst1 = cv2.resize(dst1, (300, 300))
    dst2 = cv2.resize(dst2, (300, 300))
    cv2.imwrite('C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/Translation_R/' + 'T_' + file_name, dst1)
    cv2.imwrite('C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/Translation_L/' + 'T_' + file_name, dst2)

def Gray (input, file_name):
    img = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/Gray/' + 'G_' + file_name, img)

def _main():
    count = 0
    #이미지 파일들이 있는 경로
    IMAGE_DIR_BASE = 'C:/Users/Kim hyung il/Desktop/naver/D2_CAMPUS_FEST_train/train'
    image_file_list = os.listdir(IMAGE_DIR_BASE)
    for file_name in image_file_list:
        print(count)
        img = cv2.imread(IMAGE_DIR_BASE + '/' + file_name)
        rows, cols, ch = img.shape
        Affine(img, file_name)
        Blur(img, file_name)
        Flip(img, file_name)
        Translation(img, file_name)

        count+=1

if if __name__ == "__main__":
    _main()
