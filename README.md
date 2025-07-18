# 🤖 Hand Gesture Recognition using MediaPipe + Machine Learning

A real-time hand gesture recognition system built using Python, OpenCV, MediaPipe, and a machine learning model.  
This project detects **10 unique hand signs** and classifies them with high accuracy using a Random Forest Classifier.

---

## ✨ Features

- 📸 Real-time hand tracking using MediaPipe
- 🔍 Rule-based detection (optional)
- 🧠 ML-based gesture prediction (Random Forest)
- 📁 Collect and save hand landmarks in one CSV
- 🛠 Easy-to-train custom model
- 🚀 Fully offline & works with webcam

---

## 🧠 Supported Hand Signs

| Key | Gesture Label       |
|-----|----------------------|
| 1   | Fist                |
| 2   | Open Palm           |
| 3   | Thumbs Up           |
| 4   | Peace               |
| 5   | Point Up            |
| 6   | Call Me             |
| 7   | Point Down          |
| 8   | Point Right         |
| 9   | Point Left          |

---

## 📦 Requirements

Install the required libraries:

```bash
pip install opencv-python mediapipe pandas scikit-learn joblib


# 📁 Project Structure **

HandGestureML/
├── gesture_data.csv               # Dataset of hand signs
├── gesture.py                     # Dataset of hand signs
├── model.py                       # Train and save ML model
├── predict_gesture_live.py        # Real-time gesture prediction
├── gesture_model.pkl              # Trained ML model (generated after training)
└── README.md



Run this code in this order
1. gesture.py
2. model.py
3. predict_gesture_live.py
