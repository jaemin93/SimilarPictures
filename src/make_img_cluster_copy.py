import os
import numpy as np
from config import *
import cv2

# get list of files
img_paths = os.listdir(IMG_DIR)
img_paths.sort()
img_paths = [filename for filename in img_paths if filename.endswith(IMG_EXT)]

# load predicted labels
labels_pred = np.load(os.path.join(DATA_DIR, LABELS_PRED + ".npy"))

# make result dir
if os.path.exists(CLUSTER_DIR) is False:
    os.makedirs(CLUSTER_DIR)
label_file_list = os.listdir(IMG_DIR)
label_file_list.sort()

print('Start image-label matching...')
f = open(os.path.join(DATA_DIR, LABELS_PRED + ".txt"), 'r')
n = 0
while True:
   line = f.readline()
   if not line: break
   image = cv2.imread(os.path.join(IMG_DIR,label_file_list[n]))
   file_path = os.path.join(CLUSTER_DIR,line)
   if os.path.exists(file_path) is False:
       os.makedirs(file_path)
   cv2.imwrite(file_path + '/' + label_file_list[n]  + '.jpg', image)
   n = n + 1

f.close()

print('Done')