"""
전체 프로세스를 보여주는 모듈입니다.
"""
from make_labels_true import *
from extract_features_fine_tuning import *
from feature_mapping import pred_num_cluster_challenge
from make_labels_pred import *
from extract_features import *
from evaluation import *
from visualize import *
from config import *
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v1_050_224', 'The name of the architecture to train.')
tf.app.flags.DEFINE_integer(
    'number_of_cluster', -1, 'True number of clustering')
tf.app.flags.DEFINE_string(
    'fine_tuning', '/path/to/your/pb', 'there is pb file(fine tuned) in your path.')
tf.app.flags.DEFINE_string(
    'fine_tuned_layer', 'InceptionResnetV2/Logits/AvgPool_1a_8x8/AvgPool:0', 'fine tuned layer in your model.')
    
FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    if os.path.exists(IMG_DIR):
        # make true labels by analysing image filename
        make_labels_true()

        # extract image features using MobileNet V2

        if os.path.exists(FLAGS.fine_tuning):
            extract_features_fine_tuning(FLAGS.fine_tuning, FLAGS.fine_tuned_layer)
        else:
            print('you don`t have pb! cluster model:', FLAGS.model_name)
            extract_features(FLAGS.model_name)

        # make cluster using K-Means algorithm
        number_of_cluster = FLAGS.number_of_cluster
        if number_of_cluster == -1:
            number_of_cluster = pred_num_cluster_challenge._main(IMG_DIR)
        make_labels_pred(number_of_cluster)

        # evaluate clustering result by adjusted Rand index
        evaluation(os.path.join(DATA_DIR, LABELS_TRUE + ".txt"), os.path.join(DATA_DIR, LABELS_PRED + ".txt"))

        # visualize clustering using t-SNE
        visualize()
    else:
        print("Image dir not found.")
        pass
