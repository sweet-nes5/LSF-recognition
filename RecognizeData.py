import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
with mp_hands.Hands(
    static_image_mode=True, # only static images
    max_num_hands=1, # max 2 hands detection
    min_detection_confidence=0.5) as hands:# detection confidence
        image = cv2.imread("/home/nesrine/Bureau/ProjetL/bey-gresh-plong-2021/BanqueASL/hand_0_bot_seg_1_cropped.png")
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            print("continue")
        #print landmarks on image
        print("handedness:",results.multi_handedness)
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print(f'thumb finger tip coordinates: (',
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width}, '
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height})'
                  )
            print(f'Ring finger tip coordinates: (',
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                  )
            print(f'Ring finger tip coordinates: (',
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width}, '
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height})'
                  )
            print(f'Ring finger tip coordinates: (',
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width}, '
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height})'
                  )
            print(f'Ring finger tip coordinates: (',
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width}, '
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height})'
                  )

def processImage(image_path, output_dir):
    """ Process input image and save output image to given directory. """
    image = cv2.flip(cv2.imread(image_path), 1)
    if image is None: return
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_hand_landmarks:
        return # if there are no detections, we can skip the rest of the code in this function

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,

    # flip and write output image to disk
    cv2.imwrite(f"{output_dir}/{image_path.split('/')[-1]}", cv2.flip(image, 1)))

import os
for image in os.listdir("/home/nesrine/Bureau/ProjetL/bey-gresh-plong-2021/BanqueASL"):
    processImage(f"images/{image}", "output")