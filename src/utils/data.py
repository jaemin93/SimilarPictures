import time
import os, os.path
import random
import cv2
import glob
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from aug_subdir_label import make_label_dict
import pandas as pd
import numpy as np

DIR = "C:\\Users\\iceba\\develop\\data\\dummy\\img\\naver_photos\\total3"
num_cluster = 5
def _main():
    images, labels = load_images(DIR)
    images, labels = normalise_images(images, labels)
    X_train, y_train = shuffle_data(images, labels)

    # vgg19_model = keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=(224,224,3))
    resnet50_model = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))

    # vgg19_output = covnet_transform(vgg19_model, X_train)
    # print("VGG19 flattened output has {} features".format(vgg19_output.shape[1]))

    resnet50_output = covnet_transform(resnet50_model, X_train)
    print("ResNet50 flattened output has {} features".format(resnet50_output.shape[1]))
    resnet50_pca = create_fit_PCA(resnet50_output)
    resnet50_output_pca = resnet50_pca.transform(resnet50_output)
    print("\nResNet50")
    K_resnet50_pca = create_train_kmeans(resnet50_output_pca)

    print("\nResNet50")
    G_resnet50_pca = create_train_gmm(resnet50_output_pca)

    print("\nResNet50:")
    K_resnet50 = create_train_kmeans(resnet50_output)

    k_resnet50_pred = K_resnet50_pca.predict(resnet50_output)
    resnet_cluster_count = cluster_label_count(k_resnet50_pred, y_train)

    print(resnet_cluster_count)



def cluster_label_count(clusters, labels):
    
    count = {}
    
    # Get unique clusters and labels
    unique_clusters = list(set(clusters))
    unique_labels = list(set(labels))
    
    # Create counter for each cluster/label combination and set it to 0
    for cluster in unique_clusters:
        count[cluster] = {}
        
        for label in unique_labels:
            count[cluster][label] = 0
    
    # Let's count
    for i in range(len(clusters)):
        count[clusters[i]][labels[i]] +=1
    
    cluster_df = pd.DataFrame(count)
    
    return cluster_df

def create_train_gmm(data, number_of_clusters=num_cluster):
    g = GaussianMixture(n_components=number_of_clusters, covariance_type="full", random_state=728)
    
    start=time.time()
    g.fit(data)
    end=time.time()
    
    print("Training took {} seconds".format(end-start))
    
    return g

def create_train_kmeans(data, number_of_clusters=num_cluster):
    # n_jobs is set to -1 to use all available CPU cores. This makes a big difference on an 8-core CPU
    # especially when the data size gets much bigger. #perfMatters
    
    k = KMeans(n_clusters=number_of_clusters, n_jobs=-1, random_state=728)

    # Let's do some timings to see how long it takes to train.
    start = time.time()

    # Train it up
    k.fit(data)

    # Stop the timing 
    end = time.time()

    # And see how long that took
    print("Training took {} seconds".format(end-start))
    
    return k

def pca_cumsum_plot(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

def create_fit_PCA(data, n_components=None):
    
    p = PCA(n_components=n_components, random_state=728)
    p.fit(data)
    
    return p

def covnet_transform(covnet_model, raw_images):

    # Pass our training data through the network
    pred = covnet_model.predict(raw_images)

    # Flatten the array
    flat = pred.reshape(raw_images.shape[0], -1)
    
    return flat


def shuffle_data(images, labels):

    # Set aside the testing data. We won't touch these until the very end.
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0, random_state=728)
    
    return X_train, y_train

def normalise_images(images, labels):

    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    # Normalise the images
    images /= 255
    
    return images, labels

def load_images(path):
    
    # Define empty arrays where we will store our images and labels
    images = []
    labels = []
    
    for sub_dir in os.listdir(path):  
        file_path = path + os.sep + sub_dir            
        for file in os.listdir(file_path):
            img = file_path + os.sep + file
        
            # Read the image
            image = cv2.imread(img)
            # Resize it to 224 x 224
            image = cv2.resize(image, (224,224))

            # Convert it from BGR to RGB so we can plot them later (because openCV reads images as BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(file)
            # Now we add it to our array
            images.append(image)
            labels.append(sub_dir)

    return images, labels


if __name__ == "__main__":
    _main()