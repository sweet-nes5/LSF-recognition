import glob
import numpy as np

from KmeansData import *
from HandTracker import *


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
        img = cv2.imread(image_path[i])

        if img is None:
            print("Could not read the image ", i, "\n")
        else:
            img_res = detector.hand_detection(img)
            lm_list = detector.find_position(img)

            if lm_list:
                lm_array = np.asarray(lm_list)
                lm_array = lm_array[:, 1:3]  # supprime la 1ere des 3 colonnes du tableau (indice du landmark)
                criteria_values = criterias(lm_array, criteria_nb)
                data = np.block([[data], [criteria_values]])
                print(data)

    data = data[1:, :]
    print(data)
    # cv2.imshow("kmeans alphabet", img_res)
    # k = cv2.waitKey(0)

    # applying_kmeans("../hand_draw.png")
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
