import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
import pickle
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PoseExtractor:
    """
    Extract pose landmarks and features from exercise videos with movement detection.
    """
    
    def __init__(self, confidence_threshold=0.5, movement_threshold=0.02, 
                 smoothing_window=5, min_exercise_duration=1.0):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        
        self.movement_threshold = movement_threshold
        self.smoothing_window = smoothing_window
        self.min_exercise_duration = min_exercise_duration
        
        # Key landmark indices for movement detection
        self.key_landmarks = {
            'shoulders': [11, 12],  # Left and right shoulders
            'elbows': [13, 14],     # Left and right elbows
            'wrists': [15, 16],     # Left and right wrists
            'hips': [23, 24],       # Left and right hips
            'knees': [25, 26],      # Left and right knees
            'ankles': [27, 28]      # Left and right ankles
        }
    
    def extract_pose_landmarks(self, video_path):
        """
        Extract raw pose landmarks from video.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            tuple: (landmarks_sequence, frame_timestamps, success)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None, None, False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        landmarks_sequence = []
        frame_timestamps = []
        frame_count = 0
        
        logger.debug(f"Processing video: {Path(video_path).name}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            timestamp = frame_count / fps
            frame_timestamps.append(timestamp)
            
            if results.pose_landmarks:
                # Extract landmarks as flat array
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.visibility])
                landmarks_sequence.append(landmarks)
            else:
                # No pose detected - use zero array
                landmarks_sequence.append([0.0] * 99)  # 33 landmarks × 3 values
            
            frame_count += 1
        
        cap.release()
        
        if len(landmarks_sequence) == 0:
            logger.warning(f"No pose data extracted from {video_path}")
            return None, None, False
        
        return np.array(landmarks_sequence), np.array(frame_timestamps), True
    
    def calculate_movement_intensity(self, landmarks_sequence):
        """
        Calculate movement intensity for each frame based on key landmarks.
        
        Args:
            landmarks_sequence (np.array): Raw landmark sequence (frames, 99)
            
        Returns:
            np.array: Movement intensity per frame
        """
        if len(landmarks_sequence) < 2:
            return np.zeros(len(landmarks_sequence))
        
        # Reshape to (frames, 33, 3) for easier processing
        landmarks_reshaped = landmarks_sequence.reshape(-1, 33, 3)
        
        movement_intensities = []
        
        for frame_idx in range(len(landmarks_reshaped)):
            if frame_idx == 0:
                movement_intensities.append(0.0)
                continue
            
            current_frame = landmarks_reshaped[frame_idx]
            previous_frame = landmarks_reshaped[frame_idx - 1]
            
            frame_movement = 0.0
            valid_landmarks = 0
            
            # Calculate movement for key body parts
            for body_part, landmark_indices in self.key_landmarks.items():
                for landmark_idx in landmark_indices:
                    current_landmark = current_frame[landmark_idx]
                    previous_landmark = previous_frame[landmark_idx]
                    
                    # Only consider landmarks with good visibility
                    if current_landmark[2] > 0.5 and previous_landmark[2] > 0.5:
                        # Calculate Euclidean distance between frames
                        dx = current_landmark[0] - previous_landmark[0]
                        dy = current_landmark[1] - previous_landmark[1]
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        frame_movement += distance
                        valid_landmarks += 1
            
            # Average movement across valid landmarks
            if valid_landmarks > 0:
                frame_movement /= valid_landmarks
            
            movement_intensities.append(frame_movement)
        
        return np.array(movement_intensities)
    
    def detect_exercise_boundaries(self, movement_intensities, frame_timestamps):
        """
        Detect start and end of exercise movement using movement analysis.
        
        Args:
            movement_intensities (np.array): Movement intensity per frame
            frame_timestamps (np.array): Timestamp for each frame
            
        Returns:
            tuple: (start_frame, end_frame, confidence_score)
        """
        if len(movement_intensities) < 10:
            return 0, len(movement_intensities) - 1, 0.0
        
        # Smooth movement intensities to reduce noise
        if len(movement_intensities) > self.smoothing_window:
            smoothed_movement = gaussian_filter1d(movement_intensities, sigma=1.0)
        else:
            smoothed_movement = movement_intensities.copy()
        
        # Find frames with significant movement
        active_frames = smoothed_movement > self.movement_threshold
        
        if not np.any(active_frames):
            # No significant movement detected, use middle portion
            total_frames = len(movement_intensities)
            margin = int(total_frames * 0.1)  # 10% margin
            start_frame = max(0, margin)
            end_frame = min(total_frames - 1, total_frames - margin)
            return start_frame, end_frame, 0.3
        
        # Find contiguous segments of movement
        active_segments = []
        in_segment = False
        segment_start = 0
        
        for i, is_active in enumerate(active_frames):
            if is_active and not in_segment:
                segment_start = i
                in_segment = True
            elif not is_active and in_segment:
                segment_duration = frame_timestamps[i-1] - frame_timestamps[segment_start]
                if segment_duration >= self.min_exercise_duration:
                    active_segments.append((segment_start, i-1, segment_duration))
                in_segment = False
        
        # Handle case where segment extends to end of video
        if in_segment:
            segment_duration = frame_timestamps[-1] - frame_timestamps[segment_start]
            if segment_duration >= self.min_exercise_duration:
                active_segments.append((segment_start, len(movement_intensities)-1, segment_duration))
        
        if not active_segments:
            # No valid segments found, use frames with highest movement
            movement_percentile = np.percentile(smoothed_movement, 70)
            high_movement_frames = np.where(smoothed_movement >= movement_percentile)[0]
            
            if len(high_movement_frames) > 0:
                start_frame = high_movement_frames[0]
                end_frame = high_movement_frames[-1]
                confidence = 0.5
            else:
                # Fallback: use middle 80% of video
                total_frames = len(movement_intensities)
                start_frame = int(total_frames * 0.1)
                end_frame = int(total_frames * 0.9)
                confidence = 0.2
        else:
            # Use the longest segment as the main exercise
            longest_segment = max(active_segments, key=lambda x: x[2])
            start_frame, end_frame = longest_segment[0], longest_segment[1]
            
            # Add small buffer around detected segment
            buffer_frames = int(3)  # ~0.1 second buffer at 30fps
            start_frame = max(0, start_frame - buffer_frames)
            end_frame = min(len(movement_intensities) - 1, end_frame + buffer_frames)
            
            confidence = min(1.0, longest_segment[2] / 3.0)  # Higher confidence for longer segments
        
        return start_frame, end_frame, confidence
    
    def compute_additional_features(self, landmarks_sequence):
        """
        Compute additional features from pose landmarks.
        
        Args:
            landmarks_sequence (np.array): Pose landmarks (frames, 99)
            
        Returns:
            np.array: Enhanced feature sequence
        """
        if len(landmarks_sequence) == 0:
            return landmarks_sequence
        
        # Reshape to (frames, 33, 3)
        landmarks_reshaped = landmarks_sequence.reshape(-1, 33, 3)
        enhanced_features = []
        
        for frame_landmarks in landmarks_reshaped:
            frame_features = list(landmarks_sequence[len(enhanced_features)])  # Original landmarks
            
            # Joint angles
            try:
                # Left elbow angle
                left_elbow_angle = self.calculate_angle(
                    frame_landmarks[11][:2],  # Left shoulder
                    frame_landmarks[13][:2],  # Left elbow
                    frame_landmarks[15][:2]   # Left wrist
                )
                
                # Right elbow angle
                right_elbow_angle = self.calculate_angle(
                    frame_landmarks[12][:2],  # Right shoulder
                    frame_landmarks[14][:2],  # Right elbow
                    frame_landmarks[16][:2]   # Right wrist
                )
                
                # Left knee angle
                left_knee_angle = self.calculate_angle(
                    frame_landmarks[23][:2],  # Left hip
                    frame_landmarks[25][:2],  # Left knee
                    frame_landmarks[27][:2]   # Left ankle
                )
                
                # Right knee angle
                right_knee_angle = self.calculate_angle(
                    frame_landmarks[24][:2],  # Right hip
                    frame_landmarks[26][:2],  # Right knee
                    frame_landmarks[28][:2]   # Right ankle
                )
                
                frame_features.extend([left_elbow_angle, right_elbow_angle, 
                                     left_knee_angle, right_knee_angle])
            except:
                frame_features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Distance features
            try:
                # Shoulder width
                shoulder_distance = self.calculate_distance(
                    frame_landmarks[11][:2], frame_landmarks[12][:2]
                )
                
                # Hip width
                hip_distance = self.calculate_distance(
                    frame_landmarks[23][:2], frame_landmarks[24][:2]
                )
                
                # Torso length (average shoulder to average hip)
                avg_shoulder = (frame_landmarks[11][:2] + frame_landmarks[12][:2]) / 2
                avg_hip = (frame_landmarks[23][:2] + frame_landmarks[24][:2]) / 2
                torso_length = self.calculate_distance(avg_shoulder, avg_hip)
                
                frame_features.extend([shoulder_distance, hip_distance, torso_length])
            except:
                frame_features.extend([0.0, 0.0, 0.0])
            
            enhanced_features.append(frame_features)
        
        return np.array(enhanced_features)
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points."""
        try:
            # Vectors
            v1 = np.array(point1) - np.array(point2)
            v2 = np.array(point3) - np.array(point2)
            
            # Angle calculation
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
        except:
            return 0.0
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        try:
            return np.linalg.norm(np.array(point1) - np.array(point2))
        except:
            return 0.0
    
    def process_single_video(self, video_path):
        """
        Process a single video to extract pose features with movement detection.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Processing results
        """
        # Extract raw pose landmarks
        landmarks_sequence, frame_timestamps, success = self.extract_pose_landmarks(video_path)
        
        if not success:
            return {
                'success': False,
                'error': 'Failed to extract pose landmarks',
                'filename': Path(video_path).name
            }
        
        # Calculate movement intensity
        movement_intensities = self.calculate_movement_intensity(landmarks_sequence)
        
        # Detect exercise boundaries
        start_frame, end_frame, confidence = self.detect_exercise_boundaries(
            movement_intensities, frame_timestamps
        )
        
        # Extract exercise portion
        exercise_landmarks = landmarks_sequence[start_frame:end_frame+1]
        exercise_timestamps = frame_timestamps[start_frame:end_frame+1]
        exercise_movement = movement_intensities[start_frame:end_frame+1]
        
        # Compute additional features
        enhanced_features = self.compute_additional_features(exercise_landmarks)
        
        # Calculate velocity features (frame-to-frame differences)
        if len(enhanced_features) > 1:
            velocity_features = np.diff(enhanced_features, axis=0)
            # Pad with zeros for first frame
            velocity_features = np.vstack([np.zeros((1, velocity_features.shape[1])), velocity_features])
        else:
            velocity_features = np.zeros_like(enhanced_features)
        
        return {
            'success': True,
            'filename': Path(video_path).name,
            'raw_landmarks': exercise_landmarks,
            'enhanced_features': enhanced_features,
            'velocity_features': velocity_features,
            'movement_intensities': exercise_movement,
            'timestamps': exercise_timestamps,
            'detection_info': {
                'original_frames': len(landmarks_sequence),
                'exercise_frames': len(exercise_landmarks),
                'start_frame': start_frame,
                'end_frame': end_frame,
                'confidence': confidence,
                'exercise_duration': exercise_timestamps[-1] - exercise_timestamps[0] if len(exercise_timestamps) > 0 else 0
            }
        }
    
    def process_dataset(self, standardized_video_folder, output_folder):
        """
        Process entire dataset of standardized videos.
        
        Args:
            standardized_video_folder (str): Path to standardized videos
            output_folder (str): Path to save pose features
            
        Returns:
            dict: Processing results
        """
        input_path = Path(standardized_video_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processing_results = {
            'processed_exercises': {},
            'summary': {
                'total_videos': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'average_exercise_duration': 0,
                'average_detection_confidence': 0
            },
            'failed_extractions': []
        }
        
        logger.info(f"Starting pose extraction: {standardized_video_folder} -> {output_folder}")
        
        total_duration = 0
        total_confidence = 0
        successful_count = 0
        
        for exercise_folder in input_path.iterdir():
            if not exercise_folder.is_dir():
                continue
            
            exercise_name = exercise_folder.name
            logger.info(f"Processing exercise: {exercise_name}")
            
            # Create exercise output folder
            exercise_output_path = output_path / exercise_name
            exercise_output_path.mkdir(exist_ok=True)
            
            video_files = list(exercise_folder.glob("*.mp4"))
            exercise_results = []
            
            for video_file in tqdm(video_files, desc=f"Extracting {exercise_name}"):
                processing_results['summary']['total_videos'] += 1
                
                # Process video
                result = self.process_single_video(str(video_file))
                
                if result['success']:
                    # Save pose data
                    output_filename = video_file.stem + "_pose_data.pkl"
                    output_filepath = exercise_output_path / output_filename
                    
                    with open(output_filepath, 'wb') as f:
                        pickle.dump(result, f)
                    
                    exercise_results.append({
                        'input_filename': video_file.name,
                        'output_filename': output_filename,
                        'detection_info': result['detection_info']
                    })
                    
                    processing_results['summary']['successful_extractions'] += 1
                    total_duration += result['detection_info']['exercise_duration']
                    total_confidence += result['detection_info']['confidence']
                    successful_count += 1
                    
                    logger.debug(f"Successfully processed: {video_file.name}")
                else:
                    processing_results['failed_extractions'].append({
                        'exercise': exercise_name,
                        'filename': video_file.name,
                        'error': result['error']
                    })
                    processing_results['summary']['failed_extractions'] += 1
                    logger.warning(f"Failed to process {video_file.name}: {result['error']}")
            
            processing_results['processed_exercises'][exercise_name] = exercise_results
            logger.info(f"Exercise {exercise_name} complete: {len(exercise_results)} videos processed")
        
        # Calculate averages
        if successful_count > 0:
            processing_results['summary']['average_exercise_duration'] = total_duration / successful_count
            processing_results['summary']['average_detection_confidence'] = total_confidence / successful_count
        
        return processing_results
    
    def generate_extraction_report(self, processing_results):
        """
        Generate and print extraction report.
        """
        summary = processing_results['summary']
        
        print("\n" + "="*80)
        print("POSE EXTRACTION & MOVEMENT DETECTION REPORT")
        print("="*80)
        
        print(f"\nEXTRACTION SUMMARY:")
        print(f"  Total videos processed: {summary['total_videos']}")
        print(f"  Successful extractions: {summary['successful_extractions']}")
        print(f"  Failed extractions: {summary['failed_extractions']}")
        print(f"  Success rate: {summary['successful_extractions']/summary['total_videos']*100:.1f}%")
        print(f"  Average exercise duration: {summary['average_exercise_duration']:.2f} seconds")
        print(f"  Average detection confidence: {summary['average_detection_confidence']:.2f}")
        
        print(f"\nPER-EXERCISE RESULTS:")
        for exercise_name, exercise_data in processing_results['processed_exercises'].items():
            successful = len(exercise_data)
            if successful > 0:
                avg_frames = np.mean([item['detection_info']['exercise_frames'] for item in exercise_data])
                avg_confidence = np.mean([item['detection_info']['confidence'] for item in exercise_data])
                print(f"  {exercise_name}: {successful} videos, "
                      f"avg {avg_frames:.0f} frames, confidence {avg_confidence:.2f}")
        
        if processing_results['failed_extractions']:
            print(f"\nFAILED EXTRACTIONS:")
            for item in processing_results['failed_extractions']:
                print(f"  {item['exercise']}/{item['filename']}: {item['error']}")
    
    def save_extraction_report(self, processing_results, output_file="pose_extraction_report.json"):
        """
        Save extraction results to JSON file.
        """
        with open(output_file, 'w') as f:
            json.dump(processing_results, f, indent=2)
        
        logger.info(f"Extraction report saved to: {output_file}")

def run_pose_extraction(standardized_video_folder, output_folder):
    """
    Main function to run pose extraction pipeline.
    
    Args:
        standardized_video_folder (str): Path to standardized videos from Step 2
        output_folder (str): Path to save extracted pose features
        
    Returns:
        dict: Processing results
    """
    extractor = PoseExtractor()
    
    # Process dataset
    processing_results = extractor.process_dataset(standardized_video_folder, output_folder)
    
    # Generate reports
    extractor.generate_extraction_report(processing_results)
    extractor.save_extraction_report(processing_results)
    
    return processing_results

if __name__ == "__main__":
    # Example usage
    standardized_video_folder = "/path/to/standardized/videos"
    output_folder = "/path/to/pose/features"
    
    try:
        processing_results = run_pose_extraction(
            standardized_video_folder=standardized_video_folder,
            output_folder=output_folder
        )
        
        summary = processing_results['summary']
        print(f"\nPose extraction complete!")
        print(f"Successfully processed: {summary['successful_extractions']} videos")
        print(f"Average exercise duration: {summary['average_exercise_duration']:.2f} seconds")
        print(f"Output location: {output_folder}")
        
    except Exception as e:
        logger.error(f"Error during pose extraction: {e}")