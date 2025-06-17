import cv2
import os
import numpy as np
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    """
    Analyzes video dataset for quality control and preprocessing requirements.
    """
    
    def __init__(self, target_fps=30, target_width=640, target_height=480):
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        
    def analyze_video_properties(self, video_path):
        """
        Extract properties from a single video file.
        
        Args:
            video_path (Path): Path to video file
            
        Returns:
            dict: Video properties or None if video cannot be read
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'filename': video_path.name,
            'full_path': str(video_path),
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration_seconds': duration,
            'file_size_mb': video_path.stat().st_size / (1024 * 1024),
            'needs_fps_conversion': abs(fps - self.target_fps) > 1,
            'needs_resize': width != self.target_width or height != self.target_height
        }
    
    def analyze_dataset(self, dataset_folder):
        """
        Analyze all videos in the dataset folder.
        
        Expected structure:
        dataset_folder/
        ├── exercise1/
        │   ├── video1.mp4
        │   └── video2.mp4
        └── exercise2/
            └── video1.mp4
            
        Args:
            dataset_folder (str): Path to dataset root folder
            
        Returns:
            dict: Analysis results by exercise
        """
        dataset_path = Path(dataset_folder)
        analysis_results = {}
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")
        
        logger.info(f"Analyzing videos in: {dataset_folder}")
        
        # Supported video formats
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
        
        for exercise_folder in dataset_path.iterdir():
            if not exercise_folder.is_dir():
                continue
                
            exercise_name = exercise_folder.name
            logger.info(f"Processing exercise: {exercise_name}")
            
            # Find all video files
            video_files = []
            for ext in video_extensions:
                video_files.extend(exercise_folder.glob(ext))
            
            exercise_videos = []
            valid_count = 0
            
            for video_file in video_files:
                video_info = self.analyze_video_properties(video_file)
                if video_info:
                    exercise_videos.append(video_info)
                    valid_count += 1
                    logger.debug(f"Analyzed: {video_file.name}")
                else:
                    logger.error(f"Failed to analyze: {video_file.name}")
            
            analysis_results[exercise_name] = {
                'videos': exercise_videos,
                'total_files': len(video_files),
                'valid_videos': valid_count
            }
            
            logger.info(f"Exercise {exercise_name}: {valid_count}/{len(video_files)} valid videos")
        
        return analysis_results
    
    def generate_summary_statistics(self, analysis_results):
        """
        Generate summary statistics for the dataset.
        
        Args:
            analysis_results (dict): Results from analyze_dataset
            
        Returns:
            dict: Summary statistics
        """
        total_videos = 0
        total_duration = 0
        all_durations = []
        all_fps = []
        all_resolutions = []
        conversion_stats = {'fps': 0, 'resize': 0}
        
        for exercise_name, data in analysis_results.items():
            videos = data['videos']
            total_videos += len(videos)
            
            for video in videos:
                duration = video['duration_seconds']
                total_duration += duration
                all_durations.append(duration)
                all_fps.append(video['fps'])
                all_resolutions.append((video['width'], video['height']))
                
                if video['needs_fps_conversion']:
                    conversion_stats['fps'] += 1
                if video['needs_resize']:
                    conversion_stats['resize'] += 1
        
        if not all_durations:
            return {}
        
        # Find most common resolution
        resolution_counts = {}
        for res in all_resolutions:
            resolution_counts[res] = resolution_counts.get(res, 0) + 1
        most_common_resolution = max(resolution_counts.items(), key=lambda x: x[1])
        
        return {
            'total_exercises': len(analysis_results),
            'total_videos': total_videos,
            'total_duration_minutes': total_duration / 60,
            'average_videos_per_exercise': total_videos / len(analysis_results),
            'duration_stats': {
                'min': min(all_durations),
                'max': max(all_durations),
                'mean': np.mean(all_durations),
                'std': np.std(all_durations)
            },
            'fps_stats': {
                'min': min(all_fps),
                'max': max(all_fps),
                'mean': np.mean(all_fps),
                'unique_values': list(set(all_fps))
            },
            'most_common_resolution': most_common_resolution[0],
            'conversion_requirements': conversion_stats
        }
    
    def identify_quality_issues(self, analysis_results):
        """
        Identify videos with potential quality issues.
        
        Args:
            analysis_results (dict): Results from analyze_dataset
            
        Returns:
            list: List of quality issues
        """
        issues = []
        
        for exercise_name, data in analysis_results.items():
            for video in data['videos']:
                video_issues = []
                
                # Duration checks
                if video['duration_seconds'] < 3:
                    video_issues.append("duration_too_short")
                elif video['duration_seconds'] > 30:
                    video_issues.append("duration_very_long")
                
                # FPS checks
                if video['fps'] < 15:
                    video_issues.append("fps_too_low")
                elif video['fps'] > 60:
                    video_issues.append("fps_too_high")
                
                # Resolution checks
                if video['width'] < 320 or video['height'] < 240:
                    video_issues.append("resolution_too_low")
                
                # File size checks (potential corruption indicators)
                if video['file_size_mb'] < 0.5:
                    video_issues.append("file_size_suspicious")
                
                # Frame consistency check
                expected_frames = video['duration_seconds'] * video['fps']
                if abs(video['frame_count'] - expected_frames) > video['fps']:
                    video_issues.append("frame_count_inconsistent")
                
                if video_issues:
                    issues.append({
                        'exercise': exercise_name,
                        'filename': video['filename'],
                        'full_path': video['full_path'],
                        'issues': video_issues,
                        'properties': {
                            'duration': video['duration_seconds'],
                            'fps': video['fps'],
                            'resolution': f"{video['width']}x{video['height']}",
                            'size_mb': video['file_size_mb']
                        }
                    })
        
        return issues
    
    def print_analysis_report(self, analysis_results, summary_stats, quality_issues):
        """
        Print formatted analysis report to console.
        """
        print("\n" + "="*80)
        print("VIDEO DATASET ANALYSIS REPORT")
        print("="*80)
        
        # Overall summary
        print(f"\nDATASET OVERVIEW:")
        print(f"  Total exercises: {summary_stats['total_exercises']}")
        print(f"  Total videos: {summary_stats['total_videos']}")
        print(f"  Total duration: {summary_stats['total_duration_minutes']:.1f} minutes")
        print(f"  Average videos per exercise: {summary_stats['average_videos_per_exercise']:.1f}")
        
        # Per-exercise breakdown
        print(f"\nPER-EXERCISE BREAKDOWN:")
        for exercise_name, data in analysis_results.items():
            videos = data['videos']
            if videos:
                durations = [v['duration_seconds'] for v in videos]
                print(f"  {exercise_name}: {len(videos)} videos, "
                      f"duration range: {min(durations):.1f}s - {max(durations):.1f}s")
        
        # Technical specifications
        print(f"\nTECHNICAL SPECIFICATIONS:")
        print(f"  Duration range: {summary_stats['duration_stats']['min']:.1f}s - "
              f"{summary_stats['duration_stats']['max']:.1f}s")
        print(f"  FPS range: {summary_stats['fps_stats']['min']:.1f} - "
              f"{summary_stats['fps_stats']['max']:.1f}")
        print(f"  Most common resolution: {summary_stats['most_common_resolution'][0]}x"
              f"{summary_stats['most_common_resolution'][1]}")
        
        # Conversion requirements
        conv_stats = summary_stats['conversion_requirements']
        print(f"\nCONVERSION REQUIREMENTS:")
        print(f"  Videos needing FPS conversion: {conv_stats['fps']}")
        print(f"  Videos needing resize: {conv_stats['resize']}")
        
        # Quality issues
        if quality_issues:
            print(f"\nQUALITY ISSUES DETECTED:")
            issue_counts = {}
            for issue in quality_issues:
                for issue_type in issue['issues']:
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            for issue_type, count in issue_counts.items():
                print(f"  {issue_type}: {count} videos")
                
            print(f"\nDETAILED ISSUES:")
            for issue in quality_issues:
                print(f"  {issue['exercise']}/{issue['filename']}: {', '.join(issue['issues'])}")
        else:
            print(f"\nQUALITY CHECK: No major issues detected")
    
    def save_analysis_report(self, analysis_results, summary_stats, quality_issues, 
                           output_file="dataset_analysis.json"):
        """
        Save analysis results to JSON file.
        """
        report = {
            'analysis_results': analysis_results,
            'summary_statistics': summary_stats,
            'quality_issues': quality_issues,
            'analysis_parameters': {
                'target_fps': self.target_fps,
                'target_width': self.target_width,
                'target_height': self.target_height
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis report saved to: {output_file}")
        return output_file

def run_video_analysis(dataset_folder, target_fps=30, target_width=640, target_height=480):
    """
    Main function to run complete video analysis.
    
    Args:
        dataset_folder (str): Path to dataset root folder
        target_fps (int): Target FPS for conversion
        target_width (int): Target width for resizing
        target_height (int): Target height for resizing
        
    Returns:
        tuple: (analysis_results, summary_stats, quality_issues)
    """
    analyzer = VideoAnalyzer(target_fps, target_width, target_height)
    
    # Run analysis
    analysis_results = analyzer.analyze_dataset(dataset_folder)
    summary_stats = analyzer.generate_summary_statistics(analysis_results)
    quality_issues = analyzer.identify_quality_issues(analysis_results)
    
    # Generate reports
    analyzer.print_analysis_report(analysis_results, summary_stats, quality_issues)
    report_file = analyzer.save_analysis_report(analysis_results, summary_stats, quality_issues)
    
    return analysis_results, summary_stats, quality_issues

if __name__ == "__main__":
    # Example usage
    dataset_folder = "/path/to/your/exercise/videos"
    
    try:
        analysis_results, summary_stats, quality_issues = run_video_analysis(
            dataset_folder=dataset_folder,
            target_fps=30,
            target_width=640,
            target_height=480
        )
        
        print(f"\nAnalysis complete. Found {summary_stats['total_videos']} videos "
              f"across {summary_stats['total_exercises']} exercises.")
        
        if quality_issues:
            print(f"Note: {len(quality_issues)} videos have quality issues that may need attention.")
        
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")