#!/usr/bin/env python3
"""
Step 3: Convert 2D MediaPipe Landmarks to 3D Coordinates
=======================================================

This script combines Steps 1 and 2 to:
- Detect 2D facial landmarks with MediaPipe
- Look up depth values at landmark positions
- Convert to 3D coordinates in camera space
- Visualize 3D eye landmarks for gaze estimation

Requirements:
- Steps 1 and 2 working
- All previous dependencies

Usage:
- Run script to see 2D landmarks + 3D coordinates
- Focus on eye landmarks that will be used for gaze estimation
- Press 'q' to quit
- Press '3' to toggle 3D visualization
- Press 'p' to print 3D coordinates
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import sys
import math
from collections import defaultdict

class RealSenseCamera:
    """RealSense camera from Step 1"""
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
    
    def pixel_to_3d_point(self, x, y, depth_value):
        """Convert 2D pixel + depth to 3D point in camera coordinates"""
        if depth_value == 0:
            return None
        
        depth_meters = depth_value / 1000.0  # Convert mm to meters
        
        try:
            point_3d = rs.rs2_deproject_pixel_to_point(
                self.intrinsics, [x, y], depth_meters
            )
            return np.array(point_3d)
        except Exception:
            return None
    
    def stop(self):
        if self.pipeline:
            self.pipeline.stop()

class MediaPipeFaceDetector:
    """MediaPipe face detector from Step 2"""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.KEY_LANDMARKS = {
            'left_eye_corners': [33, 133],
            'left_eye_top': [159],
            'left_eye_bottom': [145],
            'left_iris_center': [468],
            'left_iris_boundary': [469, 470, 471, 472],
            
            'right_eye_corners': [362, 263],
            'right_eye_top': [386], 
            'right_eye_bottom': [374],
            'right_iris_center': [473],
            'right_iris_boundary': [474, 475, 476, 477],
            
            'nose_tip': [1],
            'chin': [18],
            'forehead': [10]
        }
    
    def detect_landmarks(self, rgb_image):
        rgb_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, 0.0
        
        face_landmarks = results.multi_face_landmarks[0]
        h, w = rgb_image.shape[:2]
        landmarks_2d = {}
        
        for category, indices in self.KEY_LANDMARKS.items():
            landmarks_2d[category] = []
            for idx in indices:
                if idx < len(face_landmarks.landmark):
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    landmarks_2d[category].append((x, y))
        
        confidence = len([p for points in landmarks_2d.values() for p in points]) / 20.0
        return landmarks_2d, min(confidence, 1.0)

class Landmarks3DConverter:
    """Converts 2D landmarks to 3D using depth data"""
    
    def __init__(self, camera):
        self.camera = camera
        self.depth_filter_size = 3  # Size of median filter for depth smoothing
        
    def convert_landmarks_to_3d(self, landmarks_2d, depth_image):
        """
        Convert 2D landmarks to 3D coordinates using depth data
        
        Args:
            landmarks_2d: Dictionary of 2D landmark coordinates from MediaPipe
            depth_image: Aligned depth image from RealSense
            
        Returns:
            landmarks_3d: Dictionary of 3D landmark coordinates
            depth_quality: Quality metrics for the depth conversion
        """
        if landmarks_2d is None:
            return None, None
        
        landmarks_3d = {}
        depth_stats = {
            'total_points': 0,
            'valid_depth_points': 0,
            'invalid_depth_points': 0,
            'avg_depth': 0,
            'depth_range': [float('inf'), 0]
        }
        
        all_depths = []
        
        for category, points_2d in landmarks_2d.items():
            landmarks_3d[category] = []
            
            for x, y in points_2d:
                depth_stats['total_points'] += 1
                
                # Get depth value with smoothing
                depth_value = self._get_smoothed_depth(depth_image, x, y)
                
                if depth_value > 0:
                    # Convert to 3D
                    point_3d = self.camera.pixel_to_3d_point(x, y, depth_value)
                    
                    if point_3d is not None:
                        landmarks_3d[category].append(point_3d)
                        depth_stats['valid_depth_points'] += 1
                        all_depths.append(depth_value)
                        
                        # Update depth range
                        depth_stats['depth_range'][0] = min(depth_stats['depth_range'][0], depth_value)
                        depth_stats['depth_range'][1] = max(depth_stats['depth_range'][1], depth_value)
                    else:
                        landmarks_3d[category].append(None)
                        depth_stats['invalid_depth_points'] += 1
                else:
                    landmarks_3d[category].append(None)
                    depth_stats['invalid_depth_points'] += 1
        
        # Calculate average depth
        if all_depths:
            depth_stats['avg_depth'] = np.mean(all_depths)
            depth_stats['depth_std'] = np.std(all_depths)
        else:
            depth_stats['depth_range'] = [0, 0]
        
        return landmarks_3d, depth_stats
    
    def _get_smoothed_depth(self, depth_image, x, y):
        """Get depth value with median filtering to reduce noise"""
        h, w = depth_image.shape
        size = self.depth_filter_size
        
        # Collect depth values in neighborhood
        depths = []
        for dy in range(-size//2, size//2 + 1):
            for dx in range(-size//2, size//2 + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    d = depth_image[ny, nx]
                    if d > 0:  # Valid depth
                        depths.append(d)
        
        # Return median depth or zero if no valid depths
        return int(np.median(depths)) if depths else 0
    
    def extract_eye_data_3d(self, landmarks_3d):
        """
        Extract and organize 3D eye data for gaze estimation
        
        Args:
            landmarks_3d: Dictionary of 3D landmarks
            
        Returns:
            eye_data_3d: Organized eye data with centers and iris positions
        """
        if landmarks_3d is None:
            return None
        
        eye_data_3d = {}
        
        # Process left eye
        if self._has_valid_eye_data(landmarks_3d, 'left'):
            left_corners = [p for p in landmarks_3d['left_eye_corners'] if p is not None]
            left_iris = landmarks_3d['left_iris_center']
            left_iris_3d = left_iris[0] if left_iris and left_iris[0] is not None else None
            
            if len(left_corners) >= 2 and left_iris_3d is not None:
                # Calculate eye center as midpoint of corners
                eye_center = np.mean(left_corners, axis=0)
                
                eye_data_3d['left'] = {
                    'eye_center': eye_center,
                    'iris_center': left_iris_3d,
                    'corners': left_corners,
                    'valid': True
                }
        
        # Process right eye
        if self._has_valid_eye_data(landmarks_3d, 'right'):
            right_corners = [p for p in landmarks_3d['right_eye_corners'] if p is not None]
            right_iris = landmarks_3d['right_iris_center']
            right_iris_3d = right_iris[0] if right_iris and right_iris[0] is not None else None
            
            if len(right_corners) >= 2 and right_iris_3d is not None:
                eye_center = np.mean(right_corners, axis=0)
                
                eye_data_3d['right'] = {
                    'eye_center': eye_center,
                    'iris_center': right_iris_3d,
                    'corners': right_corners,
                    'valid': True
                }
        
        return eye_data_3d
    
    def _has_valid_eye_data(self, landmarks_3d, eye_side):
        """Check if we have enough valid 3D data for an eye"""
        corner_key = f'{eye_side}_eye_corners'
        iris_key = f'{eye_side}_iris_center'
        
        return (corner_key in landmarks_3d and 
                iris_key in landmarks_3d and 
                len(landmarks_3d[corner_key]) >= 2 and
                len(landmarks_3d[iris_key]) >= 1)

class Visualizer3D:
    """Visualization tools for 3D landmarks"""
    
    def __init__(self):
        self.colors = {
            'left_eye': (0, 255, 0),
            'right_eye': (255, 0, 0),
            'iris': (0, 255, 255),
            'face': (255, 255, 255)
        }
    
    def draw_2d_with_3d_info(self, image, landmarks_2d, landmarks_3d, depth_stats):
        """Draw 2D landmarks with 3D coordinate information"""
        if landmarks_2d is None:
            return image
        
        display_image = image.copy()
        
        # Draw landmarks with 3D info
        for category, points_2d in landmarks_2d.items():
            points_3d = landmarks_3d.get(category, []) if landmarks_3d else []
            
            for i, (x, y) in enumerate(points_2d):
                # Get corresponding 3D point if available
                point_3d = points_3d[i] if i < len(points_3d) and points_3d[i] is not None else None
                
                # Choose color based on 3D validity
                if point_3d is not None:
                    if 'iris' in category:
                        color = self.colors['iris']
                        size = 4
                    elif 'left' in category:
                        color = self.colors['left_eye']
                        size = 3
                    elif 'right' in category:
                        color = self.colors['right_eye']
                        size = 3
                    else:
                        color = self.colors['face']
                        size = 2
                else:
                    color = (128, 128, 128)  # Gray for invalid depth
                    size = 2
                
                cv2.circle(display_image, (x, y), size, color, -1)
                
                # Draw depth value for iris centers
                if 'iris_center' in category and point_3d is not None:
                    depth_mm = int(np.linalg.norm(point_3d) * 1000)
                    cv2.putText(display_image, f"{depth_mm}mm", (x+10, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw depth statistics
        if depth_stats:
            y_offset = 30
            stats_text = [
                f"3D Points: {depth_stats['valid_depth_points']}/{depth_stats['total_points']}",
                f"Avg Depth: {depth_stats['avg_depth']:.0f}mm",
                f"Depth Range: {depth_stats['depth_range'][0]:.0f}-{depth_stats['depth_range'][1]:.0f}mm"
            ]
            
            for text in stats_text:
                cv2.putText(display_image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
        
        return display_image
    
    def draw_3d_coordinates(self, image, eye_data_3d):
        """Draw 3D coordinate information for eyes"""
        if eye_data_3d is None:
            return image
        
        y_start = image.shape[0] - 100
        
        for i, (eye_name, eye_data) in enumerate(eye_data_3d.items()):
            if not eye_data.get('valid', False):
                continue
                
            x_start = 10 + i * 300
            y_pos = y_start
            
            # Eye header
            color = self.colors['left_eye'] if eye_name == 'left' else self.colors['right_eye']
            cv2.putText(image, f"{eye_name.upper()} EYE 3D:", (x_start, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 20
            
            # Eye center coordinates
            if 'eye_center' in eye_data:
                center = eye_data['eye_center']
                cv2.putText(image, f"Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})",
                           (x_start, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_pos += 15
            
            # Iris coordinates
            if 'iris_center' in eye_data:
                iris = eye_data['iris_center']
                cv2.putText(image, f"Iris:  ({iris[0]:.3f}, {iris[1]:.3f}, {iris[2]:.3f})",
                           (x_start, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['iris'], 1)
                y_pos += 15
            
            # Distance calculation
            if 'eye_center' in eye_data and 'iris_center' in eye_data:
                distance = np.linalg.norm(eye_data['iris_center'] - eye_data['eye_center'])
                cv2.putText(image, f"Eye-Iris: {distance*1000:.1f}mm",
                           (x_start, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return image

def main():
    print("3D Landmark Conversion - Step 3")
    print("=" * 40)
    
    # Initialize camera
    camera = RealSenseCamera()
    if not camera.initialize_camera():
        print("Failed to initialize camera. Make sure Step 1 works!")
        sys.exit(1)
    
    # Initialize face detector
    face_detector = MediaPipeFaceDetector()
    print("MediaPipe face detector initialized")
    
    # Initialize 3D converter
    landmarks_converter = Landmarks3DConverter(camera)
    print("3D converter initialized")
    
    # Initialize visualizer
    visualizer = Visualizer3D()
    
    # Setup display
    cv2.namedWindow('3D Landmarks', cv2.WINDOW_NORMAL)
    
    print("\nCamera running with 3D landmark conversion!")
    print("Instructions:")
    print("- Look at camera to see 2D + 3D landmarks")
    print("- Yellow numbers show depth in mm")
    print("- Press '3' to toggle 3D coordinate display")
    print("- Press 'p' to print detailed 3D coordinates")
    print("- Press 'q' to quit")
    print("-" * 50)
    
    show_3d_coords = True
    frame_count = 0
    
    try:
        while True:
            # Get frames
            color_image, depth_image = camera.get_frames()
            if color_image is None:
                continue
            
            frame_count += 1
            
            # Detect 2D landmarks
            landmarks_2d, confidence = face_detector.detect_landmarks(color_image)
            
            # Convert to 3D
            landmarks_3d, depth_stats = landmarks_converter.convert_landmarks_to_3d(
                landmarks_2d, depth_image
            )
            
            # Extract eye data for gaze estimation
            eye_data_3d = landmarks_converter.extract_eye_data_3d(landmarks_3d)
            
            # Visualize
            display_image = visualizer.draw_2d_with_3d_info(
                color_image, landmarks_2d, landmarks_3d, depth_stats
            )
            
            if show_3d_coords:
                display_image = visualizer.draw_3d_coordinates(display_image, eye_data_3d)
            
            # Add status info
            status_color = (0, 255, 0) if landmarks_2d is not None else (0, 0, 255)
            status_text = f"Frame {frame_count} | 3D Conversion: {'OK' if landmarks_3d else 'FAILED'}"
            cv2.putText(display_image, status_text, (10, display_image.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            cv2.imshow('3D Landmarks', display_image)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('3'):
                show_3d_coords = not show_3d_coords
                print(f"3D coordinate display: {'ON' if show_3d_coords else 'OFF'}")
            elif key == ord('p') and eye_data_3d:
                # Print detailed 3D coordinates
                print(f"\n=== Frame {frame_count} - Detailed 3D Coordinates ===")
                for eye_name, eye_data in eye_data_3d.items():
                    if eye_data.get('valid', False):
                        print(f"{eye_name.upper()} EYE:")
                        print(f"  Eye Center: {eye_data['eye_center']}")
                        print(f"  Iris Center: {eye_data['iris_center']}")
                        dist = np.linalg.norm(eye_data['iris_center'] - eye_data['eye_center'])
                        print(f"  Eye-Iris Distance: {dist*1000:.2f}mm")
                print("=" * 50)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()