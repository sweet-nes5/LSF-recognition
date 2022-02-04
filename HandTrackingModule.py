import cv2
import mediapipe as mp
import time


class HandTracker:
    def __init__(self, mode=False, max_hands=2, complexity=1, detection_con=0.5, track_con=0.5):

        self.mode = mode
        self.maxHands = max_hands
        self.complexity = complexity
        self.detectionCon = detection_con
        self.trackCon = track_con
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,self.detectionCon, self.trackCon,)

        self.mpDraw = mp.solutions.drawing_utils
    def HandDetection(self, img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results= self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks: #to check if there are more than 1 hand
            for handlms in self.results.multi_hand_landmarks :
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img
    def FindPosition(self, img,HandNum=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[HandNum]

            for id, lm in enumerate(myHand.landmark):
                height,width,c=img.shape
                cx,cy=int(lm.x*width),int (lm.y*height) #to get the location in pixels
                #print(id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0),cv2.FILLED)
        return lmList

def main():
    PreviousTime = 0
    CurrentTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandTracker()
    while True:
         success, img = cap.read()
         img= detector.HandDetection(img)
         lmList= detector.FindPosition(img)
         if len(lmList)!=0:
            print(lmList[4])

         CurrentTime = time.time()
         fps = 1 / (CurrentTime - PreviousTime)
         PreviousTime = CurrentTime
         cv2.putText(img, str(int(fps)), (10, 70)  # position of the text
                     , cv2.FONT_HERSHEY_COMPLEX  # font of the text_
                     , 3  # scale
                     , (255, 255, 255)  # color
                     , 3  # the font size
                    )
         cv2.imshow("image", img)
         cv2.waitKey(1)
if __name__ == "__main__":
    main()