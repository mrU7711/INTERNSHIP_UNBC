
import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model and label encoder
model = joblib.load('C:/Users/zouha/Desktop/INTERNSHIP_unbc/HPE/classification/data/pose_classifier.pkl')
label_encoder = joblib.load('C:/Users/zouha/Desktop/INTERNSHIP_unbc/HPE/classification/data/label_encoder.pkl')

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Initialize webcam
cap = cv2.VideoCapture(0)

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

            # Extract key joints for angle calculations
            try:
                right_elbow = [landmarks[14].x, landmarks[14].y]
                right_shoulder = [landmarks[12].x, landmarks[12].y]
                right_hip = [landmarks[24].x, landmarks[24].y]

                left_elbow = [landmarks[13].x, landmarks[13].y]
                left_shoulder = [landmarks[11].x, landmarks[11].y]
                left_hip = [landmarks[23].x, landmarks[23].y]

                right_knee = [landmarks[26].x, landmarks[26].y]
                right_ankle = [landmarks[28].x, landmarks[28].y]

                left_knee = [landmarks[25].x, landmarks[25].y]
                left_ankle = [landmarks[27].x, landmarks[27].y]

                right_wrist = [landmarks[16].x, landmarks[16].y]
                left_wrist = [landmarks[15].x, landmarks[15].y]

                # Calculate angles
                angles = [
                    calculate_angle(right_elbow, right_shoulder, right_hip),
                    calculate_angle(left_elbow, left_shoulder, left_hip),
                    calculate_angle(right_knee, right_hip, left_knee),
                    calculate_angle(right_hip, right_knee, right_ankle),
                    calculate_angle(left_hip, left_knee, left_ankle),
                    calculate_angle(right_wrist, right_elbow, right_shoulder),
                    calculate_angle(left_wrist, left_elbow, left_shoulder)
                ]

                # Predict movement
                prediction = model.predict([angles])[0]
                label = label_encoder.inverse_transform([prediction])[0]

                # Display result
                cv2.putText(image, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            except Exception as e:
                print("Angle calculation error:", e)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Movement Classifier', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
