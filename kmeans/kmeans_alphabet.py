import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from KmeansData import *
from HandTracker import *

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def drawing_clusters(data, model):
    fig = plt.figure(figsize=(4, 4))
    fig.add_subplot(111, projection='3d')

    labels = np.array(model.labels_[:])

    cmap = plt.cm.Spectral
    norm = plt.Normalize()
    plt.scatter(data[:, 0], data[:, 1], data[:, 2], marker="x", c=cmap(norm(labels)))

    # draws the center of each cluster
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], model.cluster_centers_[:, 2],
                marker="o", color="green")
    plt.show()


def criterias(landmarks, criteria_nb):
    criteria_values = np.zeros(criteria_nb)

    for i in range(1, len(landmarks)):
        criteria_values[2*i - 2] = landmarks[0][0] - landmarks[i][0]  # compare x
        criteria_values[2*i - 1] = landmarks[0][1] - landmarks[i][1]  # compare y

    return criteria_values


def main():
    image_path = glob.glob("./Alphabet/*.png")  # List all the files whose name matches "path"
    detector = HandTracker()
    criteria_nb = 40
    data = np.zeros(criteria_nb)

    for i in range(0, len(image_path)):
        print(i)
        img = cv2.imread(image_path[i])

        if img is None:
            print("Could not read the image ", i, "\n")
        else:
            # detect the hand landmarks on the image
            detector.hand_detection(img)
            lm_list = detector.find_position(img)

            if lm_list:
                # Calculates the values of the criterias that will be used to train the k-means model
                lm_array = np.asarray(lm_list)
                lm_array = lm_array[:, 1:3]  # delete the 1st of the 3 rows (the landmark indexes)
                criteria_values = criterias(lm_array, criteria_nb)

                # Adds these values to the array 'data'
                data = np.block([[data], [criteria_values]])

    '''
    # Preprocess the data
    pca = PCA(n_components=3, random_state=42)
    data_processed = pca.fit_transform(data[1:, :])
    print("data_process ", data_processed)
    '''

    # Applies the kmeans algorithm to the data
    data = data[1:, :]
    model = KMeans(n_clusters=26, init="k-means++", n_init=10, max_iter=1000)
    model.fit(data)

    # Saves the data and the k-means model obtained
    save_object(KmeansData(data, model), "cluster_alphabet_40-v02")


if __name__ == "__main__":
    main()
