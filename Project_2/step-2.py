#!/usr/bin/env python3
"""
Step 2: MediaPipe Face Detection and Eye Landmarks
==================================================

This script combines RealSense camera with MediaPipe to detect:
- Face mesh (468 landmarks)
- Iris landmarks (5 per eye)
- Specifically extract eye corners and iris centers needed for gaze estimation

Requirements:
- Step 1 working (RealSense camera)
- mediapipe library

Installation:
pip install mediapipe

Usage:
- Run script to see face landmarks in real-time
- Focus on eye region landmarks
- Press 'q' to quit
- Press 'd' to toggle detailed landmark view
- Press 'e' to show only eye landmarks
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import sys
from collections import defaultdict

class RealSenseCamera:
    """RealSense camera class from Step 1"""
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.profile = None
        self.align = None
        self.intrinsics = None
        
    def initialize_camera(self, width=640, height=480, fps=30):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            
            color_profile = self.profile.get_stream(rs.stream.color)
            self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            print("Camera initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None
    
    def stop(self):
        if self.pipeline:
            self.pipeline.stop()

class MediaPipeFaceDetector:
    """MediaPipe face detection and landmark extraction"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Face Mesh with iris tracking
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define key landmark indices for gaze estimation
        self.KEY_LANDMARKS = {
            # Left eye landmarks
            'left_eye_corners': [33, 133],  # Inner corner, outer corner
            'left_eye_top': [159],
            'left_eye_bottom': [145],
            'left_iris_center': [468],
            'left_iris_boundary': [469, 470, 471, 472],  # Top, bottom, left, right
            
            # Right eye landmarks  
            'right_eye_corners': [362, 263],  # Inner corner, outer corner
            'right_eye_top': [386], 
            'right_eye_bottom': [374],
            'right_iris_center': [473],
            'right_iris_boundary': [474, 475, 476, 477],  # Top, bottom, left, right
            
            # Face orientation landmarks
            'nose_tip': [1],
            'chin': [18],
            'forehead': [10]
        }
        
        # Colors for different landmark types
        self.COLORS = {
            'left_eye_corners': (0, 255, 0),      # Green
            'left_iris_center': (0, 255, 255),    # Yellow
            'left_iris_boundary': (0, 200, 200),  # Light cyan
            'right_eye_corners': (255, 0, 0),     # Blue  
            'right_iris_center': (255, 255, 0),   # Cyan
            'right_iris_boundary': (200, 200, 0), # Light blue
            'face_ref': (255, 255, 255)           # White
        }
    
    def detect_landmarks(self, rgb_image):
        """
        Detect facial landmarks and extract key points for gaze estimation
        
        Args:
            rgb_image: BGR image from camera
            
        Returns:
            landmarks_2d: Dictionary of 2D landmark coordinates
            detection_confidence: Confidence score of detection
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, 0.0
        
        # Get the first (and only) face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert normalized coordinates to pixel coordinates
        h, w = rgb_image.shape[:2]
        landmarks_2d = {}
        
        # Extract all key landmarks
        for category, indices in self.KEY_LANDMARKS.items():
            landmarks_2d[category] = []
            for idx in indices:
                if idx < len(face_landmarks.landmark):
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    landmarks_2d[category].append((x, y))
        
        # Calculate detection confidence (rough estimate based on landmark quality)
        confidence = self._calculate_detection_confidence(landmarks_2d, w, h)
        
        return landmarks_2d, confidence
    
    def _calculate_detection_confidence(self, landmarks_2d, width, height):
        """Calculate rough confidence score based on landmark positions"""
        try:
            # Check if key landmarks are within reasonable bounds
            score = 1.0
            
            # Check if iris centers are detected
            if not landmarks_2d.get('left_iris_center') or not landmarks_2d.get('right_iris_center'):
                score *= 0.5
            
            # Check if landmarks are within image bounds
            for category, points in landmarks_2d.items():
                for x, y in points:
                    if x < 0 or x >= width or y < 0 or y >= height:
                        score *= 0.8
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.0
    
    def visualize_landmarks(self, image, landmarks_2d, show_all=False, eye_only=False):
        """
        Draw landmarks on the image
        
        Args:
            image: Image to draw on
            landmarks_2d: Dictionary of landmark coordinates
            show_all: Show all 468 face landmarks
            eye_only: Show only eye-related landmarks
        """
        if landmarks_2d is None:
            return image
        
        # Draw key landmarks for gaze estimation
        if not eye_only:
            # Face reference points
            face_refs = ['nose_tip', 'chin', 'forehead']
            for ref in face_refs:
                if ref in landmarks_2d:
                    for x, y in landmarks_2d[ref]:
                        cv2.circle(image, (x, y), 3, self.COLORS['face_ref'], -1)
        
        # Draw eye landmarks
        eye_categories = ['left_eye_corners', 'left_iris_center', 'left_iris_boundary',
                         'right_eye_corners', 'right_iris_center', 'right_iris_boundary']
        
        for category in eye_categories:
            if category in landmarks_2d and category in self.COLORS:
                color = self.COLORS[category]
                size = 4 if 'iris_center' in category else 2
                
                for x, y in landmarks_2d[category]:
                    cv2.circle(image, (x, y), size, color, -1)
        
        # Draw eye region boxes
        self._draw_eye_regions(image, landmarks_2d)
        
        return image
    
    def _draw_eye_regions(self, image, landmarks_2d):
        """Draw bounding boxes around eye regions"""
        # Left eye region
        if 'left_eye_corners' in landmarks_2d and len(landmarks_2d['left_eye_corners']) >= 2:
            left_points = landmarks_2d['left_eye_corners']
            if 'left_eye_top' in landmarks_2d and 'left_eye_bottom' in landmarks_2d:
                left_points.extend(landmarks_2d['left_eye_top'])
                left_points.extend(landmarks_2d['left_eye_bottom'])
            
            if len(left_points) >= 2:
                x_coords = [p[0] for p in left_points]
                y_coords = [p[1] for p in left_points]
                x1, y1 = min(x_coords) - 10, min(y_coords) - 10
                x2, y2 = max(x_coords) + 10, max(y_coords) + 10
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(image, "LEFT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Right eye region  
        if 'right_eye_corners' in landmarks_2d and len(landmarks_2d['right_eye_corners']) >= 2:
            right_points = landmarks_2d['right_eye_corners']
            if 'right_eye_top' in landmarks_2d and 'right_eye_bottom' in landmarks_2d:
                right_points.extend(landmarks_2d['right_eye_top'])
                right_points.extend(landmarks_2d['right_eye_bottom'])
            
            if len(right_points) >= 2:
                x_coords = [p[0] for p in right_points]
                y_coords = [p[1] for p in right_points]
                x1, y1 = min(x_coords) - 10, min(y_coords) - 10
                x2, y2 = max(x_coords) + 10, max(y_coords) + 10
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(image, "RIGHT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    def get_eye_data_summary(self, landmarks_2d):
        """Get summary of detected eye data for debugging"""
        if landmarks_2d is None:
            return "No face detected"
        
        summary = []
        
        # Check left eye
        left_iris = landmarks_2d.get('left_iris_center', [])
        left_corners = landmarks_2d.get('left_eye_corners', [])
        summary.append(f"Left: Iris={'✓' if left_iris else '✗'}, Corners={len(left_corners)}/2")
        
        # Check right eye
        right_iris = landmarks_2d.get('right_iris_center', [])
        right_corners = landmarks_2d.get('right_eye_corners', [])
        summary.append(f"Right: Iris={'✓' if right_iris else '✗'}, Corners={len(right_corners)}/2")
        
        return " | ".join(summary)

def main():
    print("MediaPipe Face Detection - Step 2")
    print("=" * 40)
    
    # Initialize camera
    camera = RealSenseCamera()
    if not camera.initialize_camera():
        print("Failed to initialize camera. Make sure Step 1 works first!")
        sys.exit(1)
    
    # Initialize MediaPipe
    print("Initializing MediaPipe Face Mesh...")
    face_detector = MediaPipeFaceDetector()
    print("MediaPipe initialized with iris tracking enabled")
    
    # Setup display
    cv2.namedWindow('Face Landmarks', cv2.WINDOW_NORMAL)
    
    print("\nCamera is running with face detection!")
    print("Instructions:")
    print("- Look at the camera to see face landmarks")
    print("- Press 'd' to toggle detailed view")
    print("- Press 'e' to show only eye landmarks")
    print("- Press 'q' to quit")
    print("- Green = Left eye, Blue = Right eye, Yellow/Cyan = Iris")
    print("-" * 50)
    
    # Display options
    show_detailed = False
    eye_only = False
    frame_count = 0
    detection_history = []
    
    try:
        while True:
            # Get camera frames
            color_image, depth_image = camera.get_frames()
            if color_image is None:
                continue
            
            frame_count += 1
            
            # Detect face landmarks
            landmarks_2d, confidence = face_detector.detect_landmarks(color_image)
            
            # Track detection success rate
            detection_history.append(1 if landmarks_2d is not None else 0)
            if len(detection_history) > 30:  # Keep last 30 frames
                detection_history.pop(0)
            
            detection_rate = sum(detection_history) / len(detection_history) * 100
            
            # Visualize landmarks
            display_image = color_image.copy()
            display_image = face_detector.visualize_landmarks(
                display_image, landmarks_2d, show_detailed, eye_only
            )
            
            # Add info overlay
            info_y = 30
            cv2.putText(display_image, f"Frame: {frame_count}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(display_image, f"Detection Rate: {detection_rate:.1f}%", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if landmarks_2d is not None:
                info_y += 25
                cv2.putText(display_image, f"Confidence: {confidence:.2f}", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                info_y += 25
                eye_summary = face_detector.get_eye_data_summary(landmarks_2d)
                cv2.putText(display_image, eye_summary, (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                info_y += 25
                cv2.putText(display_image, "No face detected", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show mode indicators
            mode_text = []
            if show_detailed:
                mode_text.append("DETAILED")
            if eye_only:
                mode_text.append("EYES ONLY")
            if mode_text:
                cv2.putText(display_image, " | ".join(mode_text), (10, display_image.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Display
            cv2.imshow('Face Landmarks', display_image)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_detailed = not show_detailed
                print(f"Detailed view: {'ON' if show_detailed else 'OFF'}")
            elif key == ord('e'):
                eye_only = not eye_only
                print(f"Eye-only view: {'ON' if eye_only else 'OFF'}")
            elif key == ord('i') and landmarks_2d is not None:
                # Print detailed landmark info
                print(f"\nDetailed landmark info for frame {frame_count}:")
                for category, points in landmarks_2d.items():
                    print(f"  {category}: {points}")
                print(f"  Confidence: {confidence:.3f}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print(f"\nSession Statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Final detection rate: {detection_rate:.1f}%")

if __name__ == "__main__":
    main()