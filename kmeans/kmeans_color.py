import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
from KmeansData import *


def generating_data(path):
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


def filtering_data(data):
    # Removes all the white pixels and pixels with low Lightness value (HLS is very noisy for those)
    data_size = len(data)
    i = 0
    compteur_boucle = 0  # debug
    compteur_deletion = 0  # debug
    while i < data_size:  # data_size - debug
        if i % 100 == 0:  # debug
            print(i)
        if data[i, 1] > 242 or data[i, 1] < 25:  # if lightness 10% above 0 (dark) or 5% under 255 (white)
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


def clustering_data(n_clusters, data):
    # APPLIES k-means algorithm
    model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=1000)
    model.fit(data)
    return model


def drawing_clusters(data, model):
    # DRAWS the points (from the data)
    # debug : reduce the size of the data for speed (NOT ALL the points are drawn)
    size = 2000  # size = model.labels_.size to get ALL the points
    labels = np.array(model.labels_[0:size])
    # fin debug

    # the label list is mapped to colors according to a normalization of labels (between 0 an 1) and a colormap.
    cmap = plt.cm.Spectral
    norm = plt.Normalize()
    plt.scatter(data[0:size, 0], data[0:size, 1], marker="x", c=cmap(norm(labels)), s=200)
    # debug : replace 0:size by :

    # draws the center of each cluster
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker="o", color="green", s=200)
    plt.show()


def applying_kmeans(img):
    h, w, c = cv2.imread(img).shape  # height, width and colors of the first image

    # Gets all the images in the path, converts them to HLS and creates an array with the HLS values of each pixel
    data = np.array(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2HLS))

    # flattens the array to a 2D array (HLS values * number of pixels) and keeps only H and L columns
    pixels = data.flatten().reshape(h * w, 3)
    data = np.block([[pixels[:, 0]], [pixels[:, 1]]])
    data = np.transpose(data)
    print("nombre de pixels : " + str(len(data)))

    # applying the kmeans to an image
    obj = load_object("kmeans_data.pickle")
    new_array = obj.model.transform(data)
    print("Distance de chaque pixel au centre du cluster :")
    print(new_array)


def main():
    applying_kmeans("../hand_draw.png")
    '''
    path = "./Hands/*.jpg"
    data = generating_data(path)
    data = filtering_data(data)
    clusters = clustering_data(4, data)
    drawing_clusters(data, clusters)
    '''

    # SAVES the data obtained through the kmeans-algorithm
    # kmeans_data = KmeansData(model)
    # save_object(kmeans_data)


if __name__ == "__main__":
    main()
