# QTrobot Human Pose Estimation Project

This repository contains all the work completed during my internship at UNBC, focused on real-time human pose estimation (HPE) for guided human-robot interaction using QTrobot.

---

## ✅ What I’ve Done

### 📄 Research & Literature Review
- Completed a full technical report titled *State of the Art in Real-Time Human Pose Estimation for Guided Multimodal Human-Robot Interaction*.
- Added a dedicated section on **background research** and **state of the art**, identifying gaps in current systems.
- Conducted a deep literature review and filled out a detailed paper summary sheet (`paperReading.xlsx`) with:
  - Technology used
  - Type of robot (if any)
  - Type of algorithm
  - Modality, user testing, and real-world application
- Extended the sheet with recent papers and manually completed all remaining columns after verification.

### 🤖 Implementation: Human Pose Estimation

#### 🔹 MediaPipe Pose
- Achieved **30–31 FPS**
- Used **250–280 MB** of RAM
- Clear and responsive real-time skeleton tracking
- Works well on embedded systems

#### 🔹 YOLOv8-Pose
- Achieved **10–11 FPS**
- Used **490–500 MB** of RAM
- Functional but a bit heavy for responsive use in robotics

### 🧪 Code Functionality
- Live webcam-based pose estimation using both MediaPipe and YOLO
- Real-time display of:
  - Pose landmarks (skeleton)
  - FPS (frames per second)
  - RAM usage




---

##  Author

**Khalid Zouhair**  
Intern at University of Northern British Columbia (UNBC)  
Supervisor: Dr. Shruti Chandra
