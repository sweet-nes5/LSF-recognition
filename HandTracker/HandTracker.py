import cv2
import mediapipe as mp
from collections import namedtuple


class HandTracker:
    def __init__(self, mode=False, max_hands=2, complexity=1, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.maxHands = max_hands
        self.complexity = complexity
        self.detectionCon = detection_con
        self.trackCon = track_con
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon,)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = namedtuple('results', ['multi_hand_landmarks', 'multi_hand_world_landmarks', 'multi_handedness'])

    def hand_detection(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:  # to check if there are more than 1 hand
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_num=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]

            for id_landmark, lm in enumerate(my_hand.landmark):
                height, width, c = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)  # to get the location in pixels
                # print(id,cx,cy)
                lm_list.append([id_landmark, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0), cv2.FILLED)
        return lm_list
