import HandTracker as Ht
import cv2 as cv
import time
import os   # for storing images and using directories
HT = Ht

height, width = 600, 400
cap = cv.VideoCapture(0)
cap.set(3, height)  # for size
cap.set(3, width)

path = "Alphabet"
alphabet = os.listdir(path)

printList = []
for imgpath in alphabet:
    image = cv.imread(f'{path}/{imgpath}')
    printList.append(image)

pTime = 0
detector = HT.HandTracker()
fingerTips = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.hand_detection(img)
    lmList = detector.find_position(img, draw=False)
    # print(lmList)
    if len(lmList) != 0:
        fingerPos = []
        # thumb
        if lmList[fingerTips[0]][1] > lmList[fingerTips[0] - 1][1]:  # 1 refers t x
            fingerPos.append(1)
        else:
            fingerPos.append(0)

        # 4 finger
        for i in range(1, 5):
            if lmList[fingerTips[i]][2] < lmList[fingerTips[i]-2][2]:  # 2 refers t y
                fingerPos.append(1)
            else:
                fingerPos.append(0)
        for i in range(0, 5):
            if fingerPos[0] == 0 and fingerPos[1] == 1 and fingerPos[2] == 1\
                    and fingerPos[3] == 1 and fingerPos[4] == 1:
                print("C'est la lettre B de l'alphabet!")
            elif (fingerPos[0] == 1) and (fingerPos[1] == 0) and (fingerPos[2] == 0)\
                    and (fingerPos[3] == 0) and fingerPos[4] == 0:
                print("C'est la lettre A de l'alphabet!")
            elif (fingerPos[0] == 0) and (fingerPos[1] == 0) and (fingerPos[2] == 0)\
                    and (fingerPos[3] == 0) and fingerPos[4] == 0:
                print("C'est la lettre E de l'alphabet! ")
            elif (fingerPos[0] == 0) and (fingerPos[1] == 0) and (fingerPos[2] == 1)\
                    and (fingerPos[3] == 1) and fingerPos[4] == 1:
                print("C'est la lettre F de l'alphabet! ")
            elif (fingerPos[0] == 0) and (fingerPos[1] == 1) and (fingerPos[2] == 1)\
                    and (fingerPos[3] == 0) and fingerPos[4] == 0:
                print("C'est la lettre U de l'alphabet!")

    # fps
    # mesurer les valeurs pour chaque lettre , reconnaissance automatiques , cluster de points (methodes) ,
    # k-means algo pour le cluster , developper une intelligence artificielle
    # distance euclidienne et distance manathan ,apprendre avec une base de données et
    # implementer et faire l'apprentissage machine

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, f'FPS :{int(fps)}', (400, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv.imshow("SignRecognition", img)
    cv.waitKey(1)  # 1ms delay
