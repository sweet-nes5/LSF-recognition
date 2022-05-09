import time
import sys

from sklearn.preprocessing import StandardScaler

# from myHandTracker import *
from HandTracker.HandTracker import *
from kmeans.KmeansData import *
from kmeans import KmeansData

sys.modules['KmeansData'] = KmeansData

letters = ["L", "B", "C", "E", "Y", "O", "W", "G", "I", "R", "K", "F", "U", "V", "A"]
# letters = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]


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


def sign_detection(obj, img, detector, scaler):
    # img_res = tracking(img)
    img_res = detector.hand_detection(img)
    lm_list = detector.find_position(img)

    if len(lm_list) != 0:
        # Calculates the values of the criterias that will be used to train the k-means model
        lm_array = np.asarray(lm_list)
        lm_array = lm_array[:, 1:3]  # delete the 1st of the 3 rows (the landmark indexes)
        scaled_data = scaler.fit_transform(lm_array)

        criteria_values = criterias(scaled_data)

        for i in range(len(obj.model.cluster_centers_)):
            dist_cluster = distance.euclidean(obj.model.cluster_centers_[i], criteria_values)

            # if the eucl. distance between the image and the center of a cluster is less than the std deviation
            if abs(dist_cluster - obj.cluster_stats[0][i]) < obj.cluster_stats[1][i] ** (5 / 6):
                print(letters[i])
                cv2.putText(img_res, letters[i], (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
                break

    return img_res


def main():
    obj = load_object("kmeans/cluster_15_letters-v02_scaled.pickle")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    detector = HandTracker()
    scaler = StandardScaler()
    current_time = 0
    while True:
        success, img = cap.read()
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img_res = sign_detection(obj, img, detector, scaler)
        img_res, current_time = fps(img_res, current_time)
        cv2.imshow("Reconnaissance LSF", img_res)

        if cv2.waitKey(1) == ord('q'):
            # cv2.imwrite("hand_draw.png", img)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
