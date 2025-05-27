
import cv2
import mediapipe as mp
import numpy as np
import joblib
from math import sqrt

# Load trained model and label encoder
model = joblib.load('C:/Users/zouha/Desktop/INTERNSHIP_unbc/HPE/classification/pose_classifier.pkl')
label_encoder = joblib.load('C:/Users/zouha/Desktop/INTERNSHIP_unbc/HPE/classification/label_encoder.pkl')

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def calculate_distance(p1, p2):
    return sqrt(sum((np.array(p2) - np.array(p1)) ** 2))

# Distances (EXACT 16 as per training set)
distance_pairs = [
    (11, 15), (12, 16), (23, 27), (24, 28),
    (23, 15), (24, 16), (11, 27), (12, 28),
    (23, 16), (24, 15), (13, 14), (25, 26),
    (15, 16), (27, 28), (23, 27), (24, 28)
]

# Angles (7)
angle_triplets = [
    (14, 12, 24), (13, 11, 23), (26, 24, 25),
    (24, 26, 28), (23, 25, 27), (16, 14, 12), (15, 13, 11)
]

cap = cv2.VideoCapture('C:/Users/zouha/Desktop/INTERNSHIP_unbc/HPE/classification/data/pullup_vid.mp4')

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                lm = [(lm.x, lm.y, lm.z) for lm in landmarks]

                angles = [calculate_angle(lm[a], lm[b], lm[c]) for (a, b, c) in angle_triplets]
                distances = [calculate_distance(lm[i], lm[j]) for (i, j) in distance_pairs]
                raw_coords = [coord for point in lm for coord in point]

                features = angles + distances + raw_coords

                if len(features) != 122:
                    raise ValueError(f"Feature length mismatch: got {len(features)}, expected 122")

                prediction = model.predict([features])[0]
                label = label_encoder.inverse_transform([prediction])[0]

                # Show prediction probabilities
                probs = model.predict_proba([features])[0]
                prob_string = " | ".join(f"{name}: {prob:.2f}" for name, prob in zip(label_encoder.classes_, probs))
                print(f"Probabilities → {prob_string}")



                cv2.putText(image, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            except Exception as e:
                print("Feature vector error:", e)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Movement Classifier (122 Features)', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
