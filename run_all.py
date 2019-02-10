"""
전체 프로세스를 보여주는 모듈입니다.
"""
from make_labels_true import *
from extract_features import *
from make_labels_pred import *
from evaluation import *
from visualize import *
from config import *
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v2_140_224', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'number_of_cluster', '0', 'The name of the architecture to train.')
FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    if os.path.exists(IMG_DIR):
        # make true labels by analysing image filename
        make_labels_true()

        # extract image features using MobileNet V2
        extract_features(FLAGS.model_name)

        # make cluster using K-Means algorithm
        make_labels_pred(FLAGS.number_of_cluster)

        # evaluate clustering result by adjusted Rand index
        evaluation(os.path.join('C:\\Users\\iceba\\develop\\python\\naver_d2_fest_6th\\data\\dummy', LABELS_TRUE + ".txt"), os.path.join('C:\\Users\\iceba\\develop\\python\\naver_d2_fest_6th\\data\\dummy', LABELS_PRED + ".txt"))

        # visualize clustering using t-SNE
        visualize()
    else:
        print("Image dir not found.")
        pass
