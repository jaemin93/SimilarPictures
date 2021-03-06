"""
이미지 폴더의 파일을 분석하여 예측 레이블(labels_pred)을 생성하는 모듈입니다.
"""
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn import datasets
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import metrics
from config import *
import matplotlib.pyplot as plt
from feature_mapping import pred_num_cluster_challenge as pred_num_cluster

def make_labels_pred(number_of_k, eps, min_samples):
    """
    K-Means 알고리즘으로 특징 벡터를 클러스터링 하는 함수입니다.
    예측 레이블은 DATA_DIR/LABELS_PRED.npy 에 저장됩니다.
    :return: None
    """
    # load datasets
    features = np.load(os.path.join(DATA_DIR, FEATURES + ".npy"))
    num_clusters = int(number_of_k)

    # estimate number of clusters
    if int(number_of_k) == -1:
        NUM_IMGS_PER_MODEL = pred_num_cluster._main()
        num_clusters = int(len(features)/NUM_IMGS_PER_MODEL)

    print("Estimated num_clusters: %d" % num_clusters)

    # make prediction
    labels_pred = KMeans(n_clusters=num_clusters, verbose=0).fit_predict(features)
    # save predicted labels
    np.save(os.path.join(DATA_DIR, LABELS_PRED + ".npy"), labels_pred)
    np.savetxt(os.path.join(DATA_DIR, LABELS_PRED + ".tsv"), labels_pred, "%d", delimiter="\t")
    np.savetxt(os.path.join(DATA_DIR, LABELS_PRED + ".txt"), labels_pred, "%d", delimiter="\t")

    labels_pred2 = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors',
                           assign_labels='kmeans').fit_predict(features)

    np.save(os.path.join(DATA_DIR, LABELS_PRED + '2' + ".npy"), labels_pred2)
    np.savetxt(os.path.join(DATA_DIR, LABELS_PRED + '2' + ".tsv"), labels_pred2, "%d", delimiter="\t")
    np.savetxt(os.path.join(DATA_DIR, LABELS_PRED + '2' + ".txt"), labels_pred2, "%d", delimiter="\t")

    labels_pred3 = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)

    np.save(os.path.join(DATA_DIR, LABELS_PRED + '3' + ".npy"), labels_pred3)
    np.savetxt(os.path.join(DATA_DIR, LABELS_PRED + '3' + ".tsv"), labels_pred3, "%d", delimiter="\t")
    np.savetxt(os.path.join(DATA_DIR, LABELS_PRED + '3' + ".txt"), labels_pred3, "%d", delimiter="\t")


if __name__ == '__main__':
    make_labels_pred(number_of_k, eps, min_samples)