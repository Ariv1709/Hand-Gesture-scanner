import cv2
import mediapipe as mp
import pandas as pd
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
data = []

current_label = -1
recording = False

print("""Press keys 0–9 to select gesture label.
 Press 's' to START/STOP recording for the selected gesture.
 Press 'q' to quit and save.
""")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    timestamp = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if recording and current_label != -1:
                row = [timestamp, current_label]
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y])
                data.append(row)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show recording status & label
    cv2.putText(frame, f'Label: {current_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'Recording: {"YES" if recording else "NO"}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255) if recording else (100, 100, 100), 2)

    cv2.imshow("Hand Gesture Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif ord('0') <= key <= ord('9'):
        current_label = int(chr(key))
        print(f" Gesture Label set to: {current_label}")
    elif key == ord('s'):
        recording = not recording
        print(" Recording started." if recording else " Recording stopped.")

cap.release()
cv2.destroyAllWindows()
hands.close()

# Save collected data
columns = ['timestamp', 'label'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y']]
df = pd.DataFrame(data, columns=columns)
df.to_csv("gesture_dataset.csv", index=False)

print("All data saved to 'gesture_dataset.csv'.")
