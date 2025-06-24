#!/usr/bin/env python3
"""
Step 1: RealSense Camera Setup and Testing
==========================================

This script sets up the Intel RealSense D455 camera and displays:
- RGB color stream
- Depth stream (colorized)
- Real-time depth values on mouse click

Requirements:
- Intel RealSense D455 camera
- pyrealsense2 library
- OpenCV
- NumPy

Installation:
pip install pyrealsense2 opencv-python numpy

Usage:
- Run script and two windows will appear
- Click on the RGB image to see depth values at that point
- Press 'q' to quit
- Press 's' to save current frame pair
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import os

class RealSenseCamera:
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.profile = None
        self.align = None
        self.colorizer = None
        self.intrinsics = None
        
    def initialize_camera(self, width=640, height=480, fps=30):
        """Initialize RealSense camera with specified resolution and FPS"""
        try:
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable streams
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            
            print("Starting RealSense pipeline...")
            self.profile = self.pipeline.start(self.config)
            
            # Create alignment object (align depth to color)
            self.align = rs.align(rs.stream.color)
            
            # Create colorizer for depth visualization
            self.colorizer = rs.colorizer()
            
            # Get camera intrinsics
            color_profile = self.profile.get_stream(rs.stream.color)
            self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            print("Camera initialized successfully!")
            print(f"Resolution: {width}x{height} @ {fps}FPS")
            print(f"Camera intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
            print(f"Principal point: cx={self.intrinsics.ppx:.1f}, cy={self.intrinsics.ppy:.1f}")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def get_frames(self):
        """Capture and return aligned color and depth frames"""
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            
            # Align depth to color
            aligned_frames = self.align.process(frames)
            
            # Get frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None, None
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Create colorized depth for visualization
            colorized_depth = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            
            return color_image, depth_image, colorized_depth
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None, None
    
    def get_depth_at_pixel(self, depth_image, x, y):
        """Get depth value at specific pixel coordinates"""
        if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
            depth_value = depth_image[y, x]  # Depth in mm
            return depth_value
        return 0
    
    def pixel_to_3d_point(self, x, y, depth_value):
        """Convert 2D pixel + depth to 3D point in camera coordinates"""
        if depth_value == 0:
            return None
        
        # Convert depth from mm to meters
        depth_meters = depth_value / 1000.0
        
        # Deproject to 3D point
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [x, y], depth_meters
        )
        return point_3d
    
    def stop(self):
        """Stop the camera pipeline"""
        if self.pipeline:
            self.pipeline.stop()
            print("Camera stopped.")

# Global variables for mouse callback
camera = None
current_depth_image = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to display depth values on click"""
    global camera, current_depth_image
    
    if event == cv2.EVENT_LBUTTONDOWN and current_depth_image is not None:
        # Get depth at clicked point
        depth_value = camera.get_depth_at_pixel(current_depth_image, x, y)
        
        if depth_value > 0:
            # Convert to 3D point
            point_3d = camera.pixel_to_3d_point(x, y, depth_value)
            
            print(f"Clicked at pixel ({x}, {y})")
            print(f"Depth: {depth_value} mm ({depth_value/1000.0:.3f} m)")
            if point_3d:
                print(f"3D point: X={point_3d[0]:.3f}m, Y={point_3d[1]:.3f}m, Z={point_3d[2]:.3f}m")
            print("-" * 50)
        else:
            print(f"No depth data at pixel ({x}, {y})")

def main():
    global camera, current_depth_image
    
    print("RealSense Camera Setup - Step 1")
    print("=" * 40)
    
    # Initialize camera
    camera = RealSenseCamera()
    
    if not camera.initialize_camera():
        print("Failed to initialize camera. Check connection and drivers.")
        sys.exit(1)
    
    # Set up windows
    cv2.namedWindow('RGB Stream', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Stream', cv2.WINDOW_NORMAL)
    
    # Set mouse callback for RGB window
    cv2.setMouseCallback('RGB Stream', mouse_callback)
    
    print("\nCamera is running!")
    print("Instructions:")
    print("- Click on RGB image to see depth values")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    print("- Press 'i' to print camera info")
    print("-" * 50)
    
    frame_count = 0
    
    try:
        while True:
            # Get frames
            color_image, depth_image, colorized_depth = camera.get_frames()
            
            if color_image is None:
                print("Failed to get frames")
                continue
            
            # Update global depth image for mouse callback
            current_depth_image = depth_image
            
            # Add frame counter and FPS info
            frame_count += 1
            cv2.putText(color_image, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(color_image, "Click for depth info", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display depth info on depth image
            depth_stats = f"Min: {np.min(depth_image[depth_image > 0])}mm, Max: {np.max(depth_image)}mm"
            cv2.putText(colorized_depth, depth_stats, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show images
            cv2.imshow('RGB Stream', color_image)
            cv2.imshow('Depth Stream', colorized_depth)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = frame_count
                cv2.imwrite(f'rgb_frame_{timestamp}.jpg', color_image)
                cv2.imwrite(f'depth_frame_{timestamp}.png', depth_image)
                cv2.imwrite(f'depth_colorized_{timestamp}.jpg', colorized_depth)
                print(f"Saved frame {timestamp}")
            elif key == ord('i'):
                # Print camera info
                print(f"\nCamera Information:")
                print(f"Intrinsics: fx={camera.intrinsics.fx:.1f}, fy={camera.intrinsics.fy:.1f}")
                print(f"Principal point: cx={camera.intrinsics.ppx:.1f}, cy={camera.intrinsics.ppy:.1f}")
                print(f"Distortion model: {camera.intrinsics.model}")
                print(f"Frame count: {frame_count}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()