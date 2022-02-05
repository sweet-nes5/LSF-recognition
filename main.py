import time
from HandTracker import *


def main():
    previous_time = 0
    cap = cv2.VideoCapture(0)
    # detector = HandTracker()
    while True:
        success, img = cap.read()
        """
        img = detector.hand_detection(img)
        lm_list = detector.find_position(img)
        if len(lm_list) != 0:
            print(lm_list[4])
        """
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img,
                    str(int(fps)),
                    (10, 70),  # position of the text
                    cv2.FONT_HERSHEY_COMPLEX,  # font of the text
                    3,  # scale
                    (255, 255, 255),  # color
                    3  # the font size
                    )
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
