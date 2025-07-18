import cv2
import mediapipe as mp
import joblib
import numpy as np


model = joblib.load("gesture_model.pkl")


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

print(" Real-time gesture prediction started... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])  # 42 features

            # Predict gesture
            if len(landmarks) == 42:
                prediction = model.predict([landmarks])[0]
                confidence = max(model.predict_proba([landmarks])[0])

                # Show prediction
                label_text = f'Gesture: {prediction} ({confidence:.2f})'
                cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)

    cv2.imshow("Gesture Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
