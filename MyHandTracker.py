import cv2
import numpy as np


class MyHandTracker:
    def tracking(self, img):
        img_blank = np.zeros(img.shape, dtype='uint8')

        # removes colors not matching a skin color
        img_blur = cv2.GaussianBlur(img, (11, 11), cv2.BORDER_DEFAULT)  # blurs
        img_rgb = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)  # converts to RGB

        ranges = [(np.array([200, 130, 95]), np.array([255, 255, 255])),
                  (np.array([100, 95, 45]), np.array([225, 165, 160])),
                  (np.array([140, 40, 0]), np.array([185, 75, 15])),
                  (np.array([30, 0, 0]), np.array([165, 105, 110]))]

        for r in ranges:
            mask = cv2.inRange(img_rgb, r[0], r[1])
            img_mask = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
            cv2.addWeighted(img_blank, 1, img_mask, 1, 0.0, img_blank)

        img_gray = cv2.cvtColor(img_blank, cv2.COLOR_RGB2GRAY)  # converts to grayscale

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
        img_bgr = cv2.cvtColor(img_blank, cv2.COLOR_BGR2RGB)
        return img_bgr
