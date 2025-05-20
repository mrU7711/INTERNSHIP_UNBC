import cv2
import mediapipe as mp
import numpy as np
import time
import psutil
import os

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# FPS tracking
prev_time = 0

# RAM usage tracker
process = psutil.Process(os.getpid())

with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Start timer
        curr_time = time.time()

        # Convert frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Run pose detection
        results = pose.process(image_rgb)

        # Create a black background
        black_image = frame.copy()


        # Draw landmarks (skeleton only) on black image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                black_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        # Calculate FPS
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Get RAM usage
        mem_usage = process.memory_info().rss / (1024 ** 2)  # in MB

        # Display FPS
        cv2.putText(black_image, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display RAM usage
        cv2.putText(black_image, f'MEM: {mem_usage:.1f} MB', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


        # Display
        cv2.imshow('Pose Estimation (Skeleton Only)', black_image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
