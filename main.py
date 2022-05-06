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


def sign_detection(object_name, img):
    obj = load_object(object_name)
    detector = HandTracker()

    # img_res = tracking(img)
    img_res = detector.hand_detection(img)
    lm_list = detector.find_position(img)
    if len(lm_list) != 0:
        # Calculates the values of the criterias that will be used to train the k-means model
        lm_array = np.asarray(lm_list)
        lm_array = lm_array[:, 1:3]  # delete the 1st of the 3 rows (the landmark indexes)

        criteria_values = criterias(lm_array)

        for i in range(len(obj.model.cluster_centers_)):
            dist_cluster = distance.euclidean(obj.model.cluster_centers_[i], criteria_values)

            # if the eucl. distance between the image and the center of a cluster is less than the std deviation
            if abs(dist_cluster - obj.cluster_stats[0][i]) < obj.cluster_stats[1][i] ** (5 / 6):
                print(i)
                break

    return img_res


def main():
    object_name = "kmeans/cluster_15_letters-v01.pickle"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    current_time = 0
    while True:
        success, img = cap.read()
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img_res = sign_detection(object_name, img)
        img_res, current_time = fps(img_res, current_time)
        cv2.imshow("Reconnaissance LSF", img_res)

        if cv2.waitKey(1) == ord('q'):
            # cv2.imwrite("hand_draw.png", img)
            break

    cap.release()
    cv2.destroyAllWindows()
    # cv.putText(img, f'FPS :{int(fps)}', (400, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)


if __name__ == "__main__":
    main()
