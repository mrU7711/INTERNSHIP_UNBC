import cv2
import time
import psutil
import os
from ultralytics import YOLO

# Load the YOLOv8 pose model (nano version for speed)
model = YOLO("yolov8n-pose.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# FPS tracking
prev_time = 0

# Process object for memory tracking
process = psutil.Process(os.getpid())

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Start timing
    curr_time = time.time()

    # Run YOLOv8-Pose on the frame
    results = model(frame, stream=True)

    # Visualize results
    for r in results:
        annotated_frame = r.plot()

    # Calculate FPS
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Get memory usage in MB
    mem_usage = process.memory_info().rss / (1024 ** 2)

    # Display FPS
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display RAM usage
    cv2.putText(annotated_frame, f'MEM: {mem_usage:.1f} MB', (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show the result
    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
