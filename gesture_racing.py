import cv2
import mediapipe as mp
import pyautogui
import time

KEY_GAS = 'right'
KEY_BRAKE = 'left'

mp_hands_module = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils



hands = mp_hands_module.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) 

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Track current action to prevent spamming key presses
current_action = None

def count_fingers(hand_landmarks):
    """
    Returns the number of fingers that are extended (open).
    We check if the finger TIP is higher (smaller y value) than the finger PIP joint.
    """
    fingers_up = 0
    
    # Landmark IDs for tips: 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
    # Landmark IDs for PIP joints (knuckles): 6, 10, 14, 18
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    # Check the 4 main fingers (ignoring thumb for simplicity)
    for i in range(len(tips)):
        # In screen coordinates, Y decreases as you go UP.
        # So if Tip Y < Pip Y, the finger is UP.
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pips[i]].y:
            fingers_up += 1
            
    return fingers_up

print("Starting Gesture Control...")
print("Show PALM to Accelerate.")
print("Show FIST to Brake.")
print("Press 'q' to quit.")

try:
    while True:
        success, img = cap.read()
        if not success:
            break

        # Flip image horizontally so it acts like a mirror
        img = cv2.flip(img, 1)
        
        # Convert BGR image to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        action_text = "NEUTRAL"
        color = (255, 255, 0) # Cyan

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Draw the skeleton on the hand
                mp_draw.draw_landmarks(img, hand_lms, mp_hands_module.HAND_CONNECTIONS)

                # Count extended fingers
                fingers = count_fingers(hand_lms)

                # --- CONTROL LOGIC ---
                if fingers >= 4:  # Open Palm (4 or 5 fingers)
                    if current_action != "GAS":
                        pyautogui.keyUp(KEY_BRAKE) # Ensure brake is off
                        pyautogui.keyDown(KEY_GAS) # Hold gas
                        current_action = "GAS"
                    action_text = "GAS (GO!)"
                    color = (0, 255, 0) # Green

                elif fingers == 0: # Fist (0 fingers)
                    if current_action != "BRAKE":
                        pyautogui.keyUp(KEY_GAS)   # Ensure gas is off
                        pyautogui.keyDown(KEY_BRAKE) # Hold brake
                        current_action = "BRAKE"
                    action_text = "BRAKE (STOP)"
                    color = (0, 0, 255) # Red
                
                else:
                    # Ambiguous / Transition state -> Release all
                    pyautogui.keyUp(KEY_GAS)
                    pyautogui.keyUp(KEY_BRAKE)
                    current_action = "NEUTRAL"
        else:
            # No hand detected -> Release all
            if current_action is not None:
                pyautogui.keyUp(KEY_GAS)
                pyautogui.keyUp(KEY_BRAKE)
                current_action = None

        # Display status on screen
        cv2.putText(img, f"Status: {action_text}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Hill Climb Gesture Controller", img)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup: Release keys to avoid them getting stuck
    pyautogui.keyUp(KEY_GAS)
    pyautogui.keyUp(KEY_BRAKE)
    cap.release()
    cv2.destroyAllWindows()