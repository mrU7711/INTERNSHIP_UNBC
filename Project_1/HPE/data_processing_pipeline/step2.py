import cv2
import os
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoStandardizer:
    """
    Standardizes videos to consistent format for pose estimation processing.
    """
    
    def __init__(self, target_fps=30, target_width=640, target_height=480, 
                 min_duration=3.0, min_file_size_mb=0.5):
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self.min_duration = min_duration
        self.min_file_size_mb = min_file_size_mb
        
    def should_process_video(self, video_info):
        """
        Determine if a video should be processed based on quality criteria.
        
        Args:
            video_info (dict): Video information from analysis step
            
        Returns:
            tuple: (should_process, skip_reasons)
        """
        skip_reasons = []
        
        # Duration check
        if video_info['duration_seconds'] < self.min_duration:
            skip_reasons.append("duration_too_short")
            
        # File size check
        if video_info['file_size_mb'] < self.min_file_size_mb:
            skip_reasons.append("file_size_suspicious")
            
        # Basic integrity checks
        if video_info['frame_count'] <= 0:
            skip_reasons.append("no_frames")
            
        if video_info['fps'] <= 0:
            skip_reasons.append("invalid_fps")
            
        return len(skip_reasons) == 0, skip_reasons
    
    def standardize_single_video(self, input_path, output_path):
        """
        Convert a single video to standard format.
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to output standardized video
            
        Returns:
            dict: Conversion results
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            return {'success': False, 'error': 'Cannot open input video'}
        
        # Get input properties
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling for FPS conversion
        frame_step = input_fps / self.target_fps if input_fps > self.target_fps else 1
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.target_fps, 
                             (self.target_width, self.target_height))
        
        if not out.isOpened():
            cap.release()
            return {'success': False, 'error': 'Cannot create output video'}
        
        frame_count = 0
        output_frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames for FPS conversion
                if frame_count % max(1, int(frame_step)) == 0:
                    # Resize frame
                    resized_frame = cv2.resize(frame, (self.target_width, self.target_height))
                    out.write(resized_frame)
                    output_frame_count += 1
                
                frame_count += 1
        
        except Exception as e:
            cap.release()
            out.release()
            return {'success': False, 'error': f'Processing error: {str(e)}'}
        
        finally:
            cap.release()
            out.release()
        
        # Verify output file was created
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return {'success': False, 'error': 'Output file not created or empty'}
        
        # Calculate output duration
        output_duration = output_frame_count / self.target_fps
        
        return {
            'success': True,
            'input_properties': {
                'fps': input_fps,
                'resolution': f"{input_width}x{input_height}",
                'frames': total_frames,
                'duration': total_frames / input_fps if input_fps > 0 else 0
            },
            'output_properties': {
                'fps': self.target_fps,
                'resolution': f"{self.target_width}x{self.target_height}",
                'frames': output_frame_count,
                'duration': output_duration
            },
            'conversion_applied': {
                'fps_changed': abs(input_fps - self.target_fps) > 1,
                'resolution_changed': input_width != self.target_width or input_height != self.target_height
            }
        }
    
    def process_dataset(self, analysis_results, input_folder, output_folder):
        """
        Process entire dataset based on analysis results.
        
        Args:
            analysis_results (dict): Results from Step 1 analysis
            input_folder (str): Path to original dataset
            output_folder (str): Path to output standardized dataset
            
        Returns:
            dict: Processing results
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Create output directory structure
        output_path.mkdir(parents=True, exist_ok=True)
        
        processing_results = {
            'processed_exercises': {},
            'summary': {
                'total_input_videos': 0,
                'processed_videos': 0,
                'skipped_videos': 0,
                'failed_conversions': 0
            },
            'skipped_files': [],
            'failed_conversions': []
        }
        
        logger.info(f"Starting video standardization: {input_folder} -> {output_folder}")
        
        for exercise_name, exercise_data in analysis_results.items():
            logger.info(f"Processing exercise: {exercise_name}")
            
            # Create exercise output folder
            exercise_output_path = output_path / exercise_name
            exercise_output_path.mkdir(exist_ok=True)
            
            exercise_results = {
                'processed': [],
                'skipped': [],
                'failed': []
            }
            
            videos = exercise_data['videos']
            processing_results['summary']['total_input_videos'] += len(videos)
            
            # Process each video with progress bar
            for video_info in tqdm(videos, desc=f"Processing {exercise_name}"):
                input_video_path = video_info['full_path']
                filename_stem = Path(video_info['filename']).stem
                output_video_path = exercise_output_path / f"{filename_stem}_standardized.mp4"
                
                # Check if video should be processed
                should_process, skip_reasons = self.should_process_video(video_info)
                
                if not should_process:
                    logger.debug(f"Skipping {video_info['filename']}: {skip_reasons}")
                    exercise_results['skipped'].append({
                        'filename': video_info['filename'],
                        'reasons': skip_reasons
                    })
                    processing_results['skipped_files'].append({
                        'exercise': exercise_name,
                        'filename': video_info['filename'],
                        'reasons': skip_reasons
                    })
                    processing_results['summary']['skipped_videos'] += 1
                    continue
                
                # Process video
                conversion_result = self.standardize_single_video(
                    input_video_path, str(output_video_path)
                )
                
                if conversion_result['success']:
                    exercise_results['processed'].append({
                        'input_filename': video_info['filename'],
                        'output_filename': output_video_path.name,
                        'conversion_details': conversion_result
                    })
                    processing_results['summary']['processed_videos'] += 1
                    logger.debug(f"Successfully processed: {video_info['filename']}")
                else:
                    exercise_results['failed'].append({
                        'filename': video_info['filename'],
                        'error': conversion_result['error']
                    })
                    processing_results['failed_conversions'].append({
                        'exercise': exercise_name,
                        'filename': video_info['filename'],
                        'error': conversion_result['error']
                    })
                    processing_results['summary']['failed_conversions'] += 1
                    logger.warning(f"Failed to process {video_info['filename']}: {conversion_result['error']}")
            
            processing_results['processed_exercises'][exercise_name] = exercise_results
            
            processed_count = len(exercise_results['processed'])
            skipped_count = len(exercise_results['skipped'])
            failed_count = len(exercise_results['failed'])
            
            logger.info(f"Exercise {exercise_name} complete: "
                       f"{processed_count} processed, {skipped_count} skipped, {failed_count} failed")
        
        return processing_results
    
    def generate_processing_report(self, processing_results):
        """
        Generate and print processing report.
        """
        summary = processing_results['summary']
        
        print("\n" + "="*80)
        print("VIDEO STANDARDIZATION REPORT")
        print("="*80)
        
        print(f"\nPROCESSING SUMMARY:")
        print(f"  Total input videos: {summary['total_input_videos']}")
        print(f"  Successfully processed: {summary['processed_videos']}")
        print(f"  Skipped (quality issues): {summary['skipped_videos']}")
        print(f"  Failed conversions: {summary['failed_conversions']}")
        print(f"  Success rate: {summary['processed_videos']/summary['total_input_videos']*100:.1f}%")
        
        print(f"\nPER-EXERCISE RESULTS:")
        for exercise_name, exercise_data in processing_results['processed_exercises'].items():
            processed = len(exercise_data['processed'])
            skipped = len(exercise_data['skipped'])
            failed = len(exercise_data['failed'])
            total = processed + skipped + failed
            
            print(f"  {exercise_name}: {processed}/{total} processed "
                  f"({skipped} skipped, {failed} failed)")
        
        if processing_results['skipped_files']:
            print(f"\nSKIPPED FILES SUMMARY:")
            skip_reasons = {}
            for item in processing_results['skipped_files']:
                for reason in item['reasons']:
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            
            for reason, count in skip_reasons.items():
                print(f"  {reason}: {count} files")
        
        if processing_results['failed_conversions']:
            print(f"\nFAILED CONVERSIONS:")
            for item in processing_results['failed_conversions']:
                print(f"  {item['exercise']}/{item['filename']}: {item['error']}")
    
    def save_processing_report(self, processing_results, output_file="standardization_report.json"):
        """
        Save processing results to JSON file.
        """
        report = {
            'processing_results': processing_results,
            'standardization_parameters': {
                'target_fps': self.target_fps,
                'target_width': self.target_width,
                'target_height': self.target_height,
                'min_duration': self.min_duration,
                'min_file_size_mb': self.min_file_size_mb
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing report saved to: {output_file}")

def run_video_standardization(analysis_results_file, input_folder, output_folder,
                            target_fps=30, target_width=640, target_height=480):
    """
    Main function to run video standardization pipeline.
    
    Args:
        analysis_results_file (str): Path to analysis results JSON from Step 1
        input_folder (str): Path to original dataset folder
        output_folder (str): Path to output standardized dataset folder
        target_fps (int): Target FPS for standardized videos
        target_width (int): Target width for standardized videos
        target_height (int): Target height for standardized videos
        
    Returns:
        dict: Processing results
    """
    # Load analysis results from Step 1
    try:
        with open(analysis_results_file, 'r') as f:
            analysis_data = json.load(f)
        analysis_results = analysis_data['analysis_results']
    except FileNotFoundError:
        logger.error(f"Analysis results file not found: {analysis_results_file}")
        raise
    except KeyError:
        logger.error("Invalid analysis results file format")
        raise
    
    # Initialize standardizer
    standardizer = VideoStandardizer(
        target_fps=target_fps,
        target_width=target_width,
        target_height=target_height
    )
    
    # Process dataset
    processing_results = standardizer.process_dataset(
        analysis_results, input_folder, output_folder
    )
    
    # Generate reports
    standardizer.generate_processing_report(processing_results)
    standardizer.save_processing_report(processing_results)
    
    return processing_results

if __name__ == "__main__":
    
    analysis_results_file = "dataset_analysis.json"
    input_folder = "/path/to/your/exercise/videos"
    output_folder = "/path/to/standardized/videos"
    
    try:
        processing_results = run_video_standardization(
            analysis_results_file=analysis_results_file,
            input_folder=input_folder,
            output_folder=output_folder,
            target_fps=30,
            target_width=640,
            target_height=480
        )
        
        summary = processing_results['summary']
        print(f"\nStandardization complete!")
        print(f"Successfully processed: {summary['processed_videos']} videos")
        print(f"Output location: {output_folder}")
        
        if summary['skipped_videos'] > 0:
            print(f"Note: {summary['skipped_videos']} videos were skipped due to quality issues")
        
    except Exception as e:
        logger.error(f"Error during standardization: {e}")