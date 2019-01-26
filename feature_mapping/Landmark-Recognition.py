import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO
import tensorflow as tf
import tensorflow_hub as hub
from six.moves.urllib.request import urlopen

import glob
import os
from itertools import accumulate

np.random.seed(10)

IMAGE_1_URL = 'C:\\Users\\iceba\\develop\\python\\naver_d2_fest_6th\\img\\tensorflow_hub_img\\1.jpg'
IMAGE_2_URL = 'C:\\Users\\iceba\\develop\\python\\naver_d2_fest_6th\\img\\tensorflow_hub_img\\2.jpg'

IMAGE_1_JPG = 'image_1.jpg'
IMAGE_2_JPG = 'image_2.jpg'

def _main():
    download_and_resize_image(IMAGE_1_URL, IMAGE_1_JPG)
    download_and_resize_image(IMAGE_2_URL, IMAGE_2_JPG)

    #show_images([IMAGE_1_JPG, IMAGE_2_JPG])

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.FATAL)

    m = hub.Module('https://tfhub.dev/google/delf/1')

    image_placeholder = tf.placeholder(tf.float32, shape=(None, None, 3), name='input_image')

    module_inputs={
        'image': image_placeholder,
        'score_threshold': 100.0,
        'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
        'max_feature_num': 1000,
    }

    module_outputs = m(module_inputs, as_dict=True)
    image_tf = image_input_fn([IMAGE_1_JPG, IMAGE_2_JPG])

    with tf.train.MonitoredSession() as sess:
        results_dict = dict()
        for image_path in [IMAGE_1_JPG, IMAGE_2_JPG]:
            image = sess.run(image_tf)
            print('Extracting locations and descr iptors from %s' % image_path)
            results_dict[image_path] = sess.run(
                [module_outputs['locations'], module_outputs['descriptors']],
                feed_dict={image_placeholder: image})

    match_images(results_dict, IMAGE_1_JPG, IMAGE_2_JPG)




# title the images that will be processed by DELF
def download_and_resize_image(url, filename, new_width=256, new_height=256):
    if 'http' in url:   # case url
        response = urlopen(url)
        image_data = response.read()
        pil_image = Image.open(BytesIO(image_data))
    else:               # case file path
        pil_image = Image.open(url)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS) # preprocessing
    pil_image_rgb = pil_image.convert('RGB') 
    pil_image_rgb.save(filename, format='JPEG', qaulity=90)

def show_images(image_path_list):
    plt.figure()
    for i, image_path in enumerate(image_path_list):
        plt.subplot(1, len(image_path_list), i+1)
        plt.imshow(np.asarray(Image.open(image_path)))
        plt.title(image_path)
        plt.grid(False)
        plt.yticks([])
        plt.xticks([])
    plt.show()

def image_input_fn(image_files):
    filename_queue = tf.train.string_input_producer(image_files, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    return tf.image.convert_image_dtype(image_tf, tf.float32)

def match_images(results_dict, image_1_path, image_2_path):
    distance_threshold = 0.8

    # Read features
    locations_1, descriptors_1 = results_dict[image_1_path]
    num_features_1 = locations_1.shape[0]
    print("Loaded image 1's %d features" % num_features_1)
    locations_2, descriptors_2 = results_dict[image_2_path]
    num_features_2 = locations_2.shape[0]
    print("Loaded image 2's %d features" % num_features_2)

    # Find nearest-neighbor matches using a KD tree
    d1_tree = cKDTree(descriptors_1)
    _, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=distance_threshold)

    # Select feature locations for putative matches.
    locations_2_to_use = np.array([locations_2[i,] 
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i],]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

    # Perform geometric verification using RANSAC
    _, inliers = ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)
    
    # the number of inliers as the score for retrieved images
    print('Found %d inliers' % sum(inliers))

    # Visualize correspondences
    _, ax = plt.subplots()
    img_1 = mpimg.imread(image_1_path)
    img_2 = mpimg.imread(image_2_path)
    inlier_idxs = np.nonzero(inliers)[0]
    plot_matches(
        ax,
        img_1,
        img_2,
        locations_1_to_use,
        locations_2_to_use,
        np.column_stack((inlier_idxs, inlier_idxs)),
        matches_color='b'
    )
    ax.axis('off')
    ax.set_title('DELF correspondences')
    plt.show()




if __name__ == "__main__":
    _main()




