import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np

st.title(" Real-time Hand Gesture Recognition")

run = st.checkbox('Start Webcam')
model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
drawing = mp.solutions.drawing_utils

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = [coord for lm in handLms.landmark for coord in (lm.x, lm.y)]
            if len(landmarks) == 42:
                pred = model.predict([landmarks])[0]
                cv2.putText(frame, f"Gesture: {pred}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()