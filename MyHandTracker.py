import cv2


class MyHandTracker:
    def tracking(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edge_rgb = cv2.Canny(img_rgb, 100, 200)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge_gray = cv2.Canny(img_gray, 100, 200)
        x = 10
        y = 40
        cv2.circle(edge_gray, (x, y), 15, (205, 114, 101), 1)
        return edge_gray
