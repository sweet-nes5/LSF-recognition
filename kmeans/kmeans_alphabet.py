import glob

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from KmeansData import *
from HandTracker import *


# Generates the data from the images in 'path'
# Frome each image is generated a set of values (criterias) that will be used in the kmeans algorithm
def data_generation(path):
    image_path = glob.glob(path)  # List all the files whose name matches "path"
    detector = HandTracker()
    scaler = StandardScaler()
    data = np.zeros(crit_nb)
    print(len(image_path))  # debug

    # for each image of a hand
    for i in range(0, len(image_path)):
        print(i)  # debug
        img = cv2.imread(image_path[i])

        if img is None:
            print("Could not read the image ", i, "\n")
        else:
            # detect the hand landmarks on the image
            img_res = detector.hand_detection(img)
            lm_list = detector.find_position(img)

            if len(lm_list) == 0:
                print("Could not detect a hand on image ", i)
            else:
                # Calculates the values of the criterias for the current image after scaling the values
                lm_array = np.asarray(lm_list)
                lm_array = lm_array[:, 1:3]  # delete the 1st of the 3 rows (the landmark indexes)
                scaled_data = scaler.fit_transform(lm_array)
                criteria_values = criterias(scaled_data)

                # Adds these values to the array 'data'
                data = np.block([[data], [criteria_values]])

                '''
                print(image_path[i]) # debug : affiche les images avec les landmarks
                img_res = cv2.circle(img_res, (10, 10), radius=10, color=(0, 0, 255), thickness=-1)
                img_res = cv2.circle(img_res, (300, 10), radius=10, color=(0, 0, 255), thickness=-1)
                img_res = cv2.circle(img_res, (10, 100), radius=10, color=(0, 0, 255), thickness=-1)
                cv2.imshow("test", img_res)
                if cv2.waitKey(10000000) == ord('q'):
                    # cv2.imwrite("hand_draw.png", img)
                    continue  # fin debug 
                '''

    return data[1:, :]


def main():
    # Generate the data from the hand image database
    data = data_generation("./Alphabet/letter-*/*.png")

    # Applies the kmeans algorithm to the data
    nb_letters = 15
    model = KMeans(n_clusters=nb_letters, init="k-means++", n_init=10, max_iter=1000)
    model.fit(data)

    # Calculates some statistics about each cluster (distances, average distance, variance)
    list_distances, cluster_stats = stats(model, data)
    print(cluster_stats)

    # Saves the data and the k-means model obtained
    save_object(KmeansData(data, model, list_distances, cluster_stats), "cluster_15_letters-v02_scaled")


if __name__ == "__main__":
    main()
