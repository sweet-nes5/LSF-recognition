import cv2
import glob

import matplotlib.pyplot as plt
import numpy as np
from cv2 import threshold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from KmeansData import *

import sys
np.set_printoptions(threshold=sys.maxsize)


def data_generation(path):
    image_path = glob.glob(path)  # List all the files whose name matches "path"
    h, w, c = cv2.imread(image_path[0]).shape  # height, width and colors of the first image
    nb_images = 1  # range(len(image_path))

    # Gets all the images in the path, converts them to HLS and creates an array with the HLS values of each pixel
    data = np.array([np.array(cv2.cvtColor(cv2.imread(image_path[i]), cv2.COLOR_BGR2HLS)) for i in range(nb_images)])

    # flattens the array to a 2D array (HLS values * number of pixels) and keeps only H and L columns
    pixels = data.flatten().reshape(nb_images * h * w, 3)
    data = np.block([[pixels[:, 0]], [pixels[:, 1]]])
    data = np.transpose(data)
    print("nombre de pixels : " + str(len(data)))

    # Removes all the white pixels and pixels with low Lightness value (HLS is very noisy for those)
    data_size = len(data)
    i = 0
    compteur_boucle = 0  # debug
    compteur_deletion = 0  # debug
    while i < data_size:  # data_size - debug
        if i % 100 == 0:  # debug
            print(i)
        if data[i, 1] > 242 or data[i, 1] < 25:  # lightness 10% above 0 (dark) or 5% under 255 (white)
            data = np.delete(data, i, 0)
            i = i - 1
            data_size = data_size - 1
            compteur_deletion = compteur_deletion + 1
        i = i + 1
        compteur_boucle = compteur_boucle + 1
    print("data après boucle : ")
    print(data[:100, :])
    print("nombre de pixels supprimes : " + str(compteur_deletion))
    print("nombre de tours de boucle : " + str(compteur_boucle))
    print("nombre de pixels non supprimes : " + str(data_size))

    # feature scaling (normalizes data so that they have the same weight : mean of 0 and standard deviation of 1)
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(data)

    # print(data, file=open("kmeans_data.txt", "w"))
    return data


def kmeans(n_clusters, data):
    # applies k-means algorithm
    # init="k-means++" :  ensure centroids are initialized with some distance between them (usually an improvement
    # over "random").
    # n_init : increased to ensure we find a stable solution (10 by default).
    # max_iter : increased to ensure that k-means will converge.
    model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=1000)
    model.fit(data)
    kmeans_data = KmeansData(model)
    save_object(kmeans_data)

    # print(model.labels_, file=open("kmeans_labels.txt", "w"))
    # label of each pixel (i.e. which cluster it belongs to)
    print("nombre d'iterations : " + str(model.n_iter_))

    # draws the points (from the data)
    # debug : reduce the size of the data for speed (NOT ALL the points are drawn)
    size = model.labels_.size  # size = model.labels_.size to get ALL the points
    labels = np.array(model.labels_[0:size])
    # fin debug

    # the label list is mapped to colors according to a normalization of labels (between 0 an 1) and a colormap.
    cmap = plt.cm.Spectral
    norm = plt.Normalize()
    plt.scatter(data[0:size, 0], data[0:size, 1], marker="x", c=cmap(norm(labels)), s=200)
    # debug : replace 0:size by :

    # draws the center of each cluster
    print(model.cluster_centers_)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker="o", color="green", s=200)
    plt.show()


def main():
    path = "./Hands/*.jpg"
    data = data_generation(path)
    kmeans(4, data)


if __name__ == "__main__":
    main()
