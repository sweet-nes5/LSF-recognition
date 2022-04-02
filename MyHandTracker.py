import cv2
import numpy as np


def processing(img):
    blur = cv2.medianBlur(img, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilated = cv2.dilate(blur, kernel)
    return dilated


# Filters the pixels of the image by color
# input: img must be in HLS
def color_masking(img):
    img_blank = np.zeros(img.shape, dtype='uint8')
    mask = cv2.inRange(img, np.array([0, 50, 50]), np.array([40, 255, 255]))

    img_mask = cv2.bitwise_and(img, img, mask=mask)
    cv2.addWeighted(img_blank, 1, img_mask, 1, 0.0, img_blank)
    return img_blank


# input: img must be in grayscale
def edges(img):
    img_blur = cv2.GaussianBlur(img, (11, 11), cv2.BORDER_DEFAULT)  # blurs

    # finding egdes (using canny)
    med_val = np.median(img)
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    img_edge = cv2.Canny(img_blur, lower, upper)  # finds edges
    img_dilated = cv2.dilate(img_edge, (5, 5), iterations=3)  # dilates
    contours, hierarchies = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # finding contours (using threshold)
    # ret, img_thresh = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)
    # contours, hierarchies = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def tracking(img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  # converts to HLS
    img_processed = processing(img_hls)
    img_masked = color_masking(img_processed)

    '''
    img_bgr = cv2.cvtColor(img_processed, cv2.COLOR_HSV2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # converts to grayscale
    contours = edges(img_gray)
    img_blank = np.zeros(img.shape, dtype='uint8')
    cv2.drawContours(img_blank, contours, -1, (0, 0, 255), 1)
    '''

    img_bgr = cv2.cvtColor(img_masked, cv2.COLOR_HSV2BGR)  # converts to BGR
    return img_bgr
