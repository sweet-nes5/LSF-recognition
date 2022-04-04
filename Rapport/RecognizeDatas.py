import cv2
import mediapipe as mp
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
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # flip and write output image to disk
    cv2.imwrite(f"{output_dir}/{image_path.split('/')[-1]}", cv2.flip(image, 1))