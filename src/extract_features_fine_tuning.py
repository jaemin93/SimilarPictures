
# -*- coding: utf-8 -*-
"""
MobileNet V2으로 이미지 특징 벡터를 추출하는 모듈입니다.
"""
import tensorflow as tf
from config import *
from models.model import *
import os
from tensorflow.python.platform import gfile
import numpy as np
import sys

def get_bottleneck_data(session, image_data, jpeg_data_tensor, decoded_image_tensor, bottleneck, image):
    resized_input_values = session.run(decoded_image_tensor,
                                       {jpeg_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = session.run(bottleneck,
                                    {image: resized_input_values})
    bottleneck_data = np.squeeze(bottleneck_values)
    return bottleneck_data


def add_jpeg_decoding():
  """ Adds operations that perform JPEG decoding and resizing to the graph.. """
  jpeg_data = tf.placeholder(tf.string,name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=3)
  # Convert from full range of uint8 to range [0,1] of float32.
  decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([224, 224]) #input_height, input_width
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  return jpeg_data, resized_image


def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def extract_features_fine_tuning(pb_path, fine_tuned_layer, model_name):
    """
    IMG_DIR에 있는 모든 이미지에 대해 model_name 특징 벡터를 추출합니다.
    추출된 특징 벡터는 DATA_DIR/FEATURES.npy 에 저장됩니다.
    BATCH_SIZE로 배치 사이즈를 조절할 수 있습니다.
    :return: 없음
    """

    model_path = pb_path
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_defnition = tf.GraphDef()
        graph_defnition.ParseFromString(f.read())


    bottleneck, image = (
        tf.import_graph_def(
            graph_defnition,
            name='',
            return_elements=[fine_tuned_layer,
                            'fifo_queue_Dequeue:0'])
    )

    # get list all images
    img_paths = os.listdir(IMG_DIR)
    img_paths.sort()
    
    # build dnn model
    model = Network_Model(model_name)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    features = np.ndarray(shape=[0,model.output_size])
    cnt = 0
    print("==== Make image feature data ====")
    for img in img_paths:
        cnt = cnt + 1
        printProgress(cnt, len(img_paths), 'Progress:', 'Complete', 1, 50)
        
        img_path = os.path.join(IMG_DIR, img)
        image_data = gfile.FastGFile(img_path, 'rb').read()
        # get batch of encoded images
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding()
        # get batch of features
        feature_data = get_bottleneck_data(sess, image_data, jpeg_data_tensor, decoded_image_tensor, bottleneck, image)
        feature_data = feature_data.reshape(1,model.output_size)
        features = np.concatenate((features, feature_data))
    # save npy and tsv files
    if os.path.exists(DATA_DIR) is False:
        os.makedirs(DATA_DIR)
    np.save(os.path.join(DATA_DIR, FEATURES + ".npy"), features)
    np.savetxt(os.path.join(DATA_DIR, FEATURES + ".tsv"), features, delimiter="\t")


if __name__ == '__main__':
    extract_features_fine_tuning(pb_path, fine_tuned_layer, model_name)
