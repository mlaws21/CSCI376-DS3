import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Path to the gesture recognition model
model_path = "gesture_recognizer.task"

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)

gesture_recognizer = GestureRecognizer.create_from_options(options)

def is_thumb_left(hand_landmarks, offset=0.05):
    
    thumb_tip = 4  
    thumb_base = 2
    pinky_base = 17

    return hand_landmarks.landmark[thumb_tip].x + offset < hand_landmarks.landmark[thumb_base].x and hand_landmarks.landmark[thumb_base].x < hand_landmarks.landmark[pinky_base].x

def is_thumb_right(hand_landmarks, offset=0.05):
    
    thumb_tip = 4  
    thumb_base = 2
    pinky_base = 17

    return hand_landmarks.landmark[thumb_tip].x - offset > hand_landmarks.landmark[thumb_base].x and hand_landmarks.landmark[thumb_base].x > hand_landmarks.landmark[pinky_base].x

def count_fingers(hand_landmarks, offset=0.1):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    finger_bases = [5, 9, 13, 17]  # knuckles
    fingers_up = 0

    for tip, base in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip].y + offset < hand_landmarks.landmark[base].y:
            fingers_up += 1

    return fingers_up  

def main():
    # Initialize video capture
    # NOTE: 0 gets phone 1 gets webcam
    cap = cv2.VideoCapture(1)  # 0 is the default webcam

    # Flag to make sure screen is clear before recording gestures
    can_record = True
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally and convert BGR to RGB
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Perform gesture recognition (canned gestures)
            result = gesture_recognizer.recognize(mp_image)

            # Process hand landmarks for custom gestures
            results = hands.process(image_rgb)

            # set default results of gesture detection
            fingers_up = 0
            thumb_left = False
            thumb_right = False

            
            if results.multi_hand_landmarks:
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    fingers_up = count_fingers(hand_landmarks)
                    thumb_left = is_thumb_left(hand_landmarks)
                    thumb_right = is_thumb_right(hand_landmarks)

            else:
                can_record = True
            
            # print(fingers_up)
            # Handle canned gestures
            if can_record:
                
                if result.gestures:
                    prev_can_record = can_record
                    can_record = False
                    recognized_gesture = result.gestures[0][0].category_name
                    confidence = result.gestures[0][0].score

                    # Example actions
                    if recognized_gesture == "Thumb_Up":# and confidence > 0.6:
                        gesture_text = f"{recognized_gesture} ({confidence:.2f})"
                        
                        # print("w")
                        pyautogui.press("w")
                    elif recognized_gesture == "Thumb_Down":# and confidence > 0.6:
                        gesture_text = f"{recognized_gesture} ({confidence:.2f})"

                        # print("s")
                        pyautogui.press("s")
                    elif recognized_gesture == "Open_Palm":# and confidence > 0.6:
                        gesture_text = f"{recognized_gesture} ({confidence:.2f})"
                        
                        pyautogui.press("n")
                        # NOTE: force user to click enter as a design principal thing

                    # Custom gestures for controlling the board
                    elif thumb_left:
                            gesture_text = f"thumb left"
                            pyautogui.press("a")
                            
                    elif thumb_right:
                            gesture_text = f"thumb right"
                            pyautogui.press("d")
                            
                            
                    # If we dont see a gesture to control the board then we should look for how many fingers are up
                    elif fingers_up > 0:

                            pyautogui.press(f"{fingers_up}")
                            gesture_text = f"{fingers_up} Fingers Up"
                    
                    else:
                        can_record = prev_can_record
                        
                else:
                    # lets user know nothing is being detected
                    gesture_text = "No Gesture Detected"
            else:
                # lets user know to clear the screen
                gesture_text = "Clear Screen before proceeding"

            # Display gesture recognition results
            cv2.putText(image, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
