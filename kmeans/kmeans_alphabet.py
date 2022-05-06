import glob

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance

from KmeansData import *
from HandTracker import *


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


def main():
    image_path = glob.glob("./Alphabet/letter-*/*.png")  # List all the files whose name matches "path"
    detector = HandTracker()
    data = np.zeros(crit_nb)
    print(len(image_path))

    for i in range(0, len(image_path)):
        print(i)
        img = cv2.imread(image_path[i])

        if img is None:
            print("Could not read the image ", i, "\n")
        else:
            # detect the hand landmarks on the image
            img_res = detector.hand_detection(img)
            lm_list = detector.find_position(img)

            if len(lm_list) == 0:
                print("Could not detect a hand on image ", i, "\n")
            else:
                # Calculates the values of the criterias that will be used to train the k-means model
                lm_array = np.asarray(lm_list)
                lm_array = lm_array[:, 1:3]  # delete the 1st of the 3 rows (the landmark indexes)
                criteria_values = criterias(lm_array)

                # Adds these values to the array 'data'
                data = np.block([[data], [criteria_values]])

                # debug : affiche les images avec les landmarks
                '''
                img_res = cv2.circle(img_res, (10, 10), radius=10, color=(0, 0, 255), thickness=-1)
                img_res = cv2.circle(img_res, (300, 10), radius=10, color=(0, 0, 255), thickness=-1)
                img_res = cv2.circle(img_res, (10, 100), radius=10, color=(0, 0, 255), thickness=-1)
                cv2.imshow("test", img_res)
                if cv2.waitKey(10000000) == ord('q'):
                    # cv2.imwrite("hand_draw.png", img)
                    continue
                '''
                # fin debug

    # Applies the kmeans algorithm to the data
    data = data[1:, :]
    model = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=1000)
    model.fit(data)

    # Saves the data and the k-means model obtained
    save_object(KmeansData(data, model), "cluster_3_letters-v01")


if __name__ == "__main__":
    main()
