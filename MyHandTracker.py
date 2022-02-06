import cv2
import numpy as np


class MyHandTracker:
    def tracking(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts image to grayscale

        # using edges (canny)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), cv2.BORDER_DEFAULT)  # blurs the image
        med_val = np.median(img)
        lower = int(max(0, 0.7 * med_val))
        upper = int(min(255, 1.3 * med_val))
        img_edge = cv2.Canny(img_blur, lower, upper)  # finds the image edges
        img_dilated = cv2.dilate(img_edge, (5, 5), iterations=3)  # dilates the image
        contours, hierarchies = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # using threshold
        # ret, img_thresh = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)
        # contours, hierarchies = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_blank = np.zeros(img.shape, dtype='uint8')
        cv2.drawContours(img_blank, contours, -1, (0, 0, 255), 1)  # draws all the contours on a blank image
        return img_blank
