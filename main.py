import time
import sys

# from myHandTracker import *
from HandTracker import *
from kmeans.KmeansData import *
from kmeans import KmeansData

sys.modules['KmeansData'] = KmeansData


def fps(img, previous_time):
    current_time = time.time()
    fps_number = 1 / (current_time - previous_time)
    cv2.putText(img,
                str(int(fps_number)),
                (10, 70),  # position of the text
                cv2.FONT_HERSHEY_COMPLEX,  # font of the text
                3,  # scale
                (255, 255, 255),  # color
                3  # the font size
                )
    return img, current_time


def main():
    obj = load_object("kmeans/cluster_3_letters-v01.pickle")

    # for each cluster, creates an array containing the distances between the cluster and its members
    cluster_nb = len(obj.model.cluster_centers_)
    list_distances = [[0.0] for i in range(cluster_nb)]

    for i in range(0, len(obj.model.labels_)):
        cluster = obj.model.labels_[i]
        list_distances[cluster].append(distance.euclidean(obj.model.cluster_centers_[cluster], obj.data[i]))

    # for each cluster, calculates the average and variance of the distance between the cluster and its members
    average = np.zeros((2, cluster_nb))

    for i in range(cluster_nb):
        list_distances[i].pop(0)
        print(len(list_distances[i]))
        average[0][i] = np.average(list_distances[i])
        average[1][i] = np.std(list_distances[i])

    print(average)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    current_time = 0
    detector = HandTracker()
    while True:
        success, img = cap.read()
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # img_res = tracking(img)
        img_res = detector.hand_detection(img)
        lm_list = detector.find_position(img)
        if len(lm_list) != 0:
            # Calculates the values of the criterias that will be used to train the k-means model
            lm_array = np.asarray(lm_list)
            lm_array = lm_array[:, 1:3]  # delete the 1st of the 3 rows (the landmark indexes)

            criteria_values = criterias(lm_array)

            for i in range(cluster_nb):
                dist_cluster = distance.euclidean(obj.model.cluster_centers_[i], criteria_values)

                if abs(dist_cluster - average[0][i]) < average[1][i]:
                    print(i)

        img_res, current_time = fps(img_res, current_time)
        cv2.imshow("Reconnaissance LSF", img_res)
        if cv2.waitKey(1) == ord('q'):
            # cv2.imwrite("hand_draw.png", img)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
