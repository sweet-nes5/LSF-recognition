import cv2
import numpy as np


class MyHandTracker:
    def tracking(self, img):
        img_blank = np.zeros(img.shape, dtype='uint8')

        # removes colors not matching a skin color
        img_blur = cv2.GaussianBlur(img, (11, 11), cv2.BORDER_DEFAULT)  # blurs
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(img_hsv, np.array([0, 50, 50]), np.array([40, 255, 255]))
        img_mask = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
        cv2.addWeighted(img_blank, 1, img_mask, 1, 0.0, img_blank)

        img_bgr = cv2.cvtColor(img_blank, cv2.COLOR_HSV2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # converts to grayscale

        # finding egdes (using canny)
        img_blur = cv2.GaussianBlur(img_gray, (11, 11), cv2.BORDER_DEFAULT)  # blurs
        med_val = np.median(img)
        lower = int(max(0, 0.7 * med_val))
        upper = int(min(255, 1.3 * med_val))
        img_edge = cv2.Canny(img_blur, lower, upper)  # finds edges
        img_dilated = cv2.dilate(img_edge, (5, 5), iterations=3)  # dilates
        contours, hierarchies = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # finding contours (using threshold)
        # ret, img_thresh = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)
        # contours, hierarchies = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # img_blank = np.zeros(img.shape, dtype='uint8')  # tips : remove this line to display the colors + edges
        cv2.drawContours(img_blank, contours, -1, (0, 0, 255), 1)  # draws the contours
        img_bgr = cv2.cvtColor(img_blank, cv2.COLOR_HSV2BGR)
        return img_bgr
