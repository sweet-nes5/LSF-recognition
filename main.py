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
    obj = load_object("kmeans/kmeans_data.pickle")
    print("distance moyenne aux clusters : ")
    print(obj.model.inertia_)

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
            print(lm_list[:])

        img_res, current_time = fps(img_res, current_time)
        cv2.imshow("Reconnaissance LSF", img_res)
        if cv2.waitKey(1) == ord('q'):
            # cv2.imwrite("hand_draw.png", img)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
