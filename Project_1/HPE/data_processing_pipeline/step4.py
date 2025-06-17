import numpy as np
import pickle
from pathlib import Path
import json
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from scipy.interpolate import interp1d
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SequencePreparator:
    """
    Prepare training sequences from pose features with data augmentation.
    """
    
    def __init__(self, sequence_length=30, overlap_ratio=0.5, min_confidence=0.3,
                 min_sequence_frames=20, augmentation_factor=3):
        self.sequence_length = sequence_length  # 30 frames = 1 second at 30 FPS
        self.overlap_ratio = overlap_ratio      # 50% overlap between sequences
        self.min_confidence = min_confidence    # Minimum detection confidence
        self.min_sequence_frames = min_sequence_frames  # Minimum frames for valid sequence
        self.augmentation_factor = augmentation_factor  # How many augmented versions per sequence
        
        # Exercise class mapping
        self.exercise_classes = {}
        self.class_to_idx = {}
        
    def load_pose_data(self, pose_features_folder):
        """
        Load all pose feature files from the folder structure.
        
        Args:
            pose_features_folder (str): Path to pose features from Step 3
            
        Returns:
            dict: Loaded pose data organized by exercise
        """
        features_path = Path(pose_features_folder)
        pose_data = {}
        
        logger.info(f"Loading pose data from: {pose_features_folder}")
        
        for exercise_folder in features_path.iterdir():
            if not exercise_folder.is_dir():
                continue
                
            exercise_name = exercise_folder.name
            exercise_files = list(exercise_folder.glob("*_pose_data.pkl"))
            
            logger.info(f"Loading {len(exercise_files)} files for exercise: {exercise_name}")
            
            exercise_data = []
            successful_loads = 0
            
            for pose_file in exercise_files:
                try:
                    with open(pose_file, 'rb') as f:
                        pose_info = pickle.load(f)
                    
                    # Quality check
                    if (pose_info['success'] and 
                        pose_info['detection_info']['confidence'] >= self.min_confidence and
                        len(pose_info['enhanced_features']) >= self.min_sequence_frames):
                        
                        exercise_data.append({
                            'filename': pose_file.name,
                            'features': pose_info['enhanced_features'],
                            'velocity_features': pose_info['velocity_features'],
                            'movement_intensities': pose_info['movement_intensities'],
                            'detection_info': pose_info['detection_info']
                        })
                        successful_loads += 1
                    else:
                        logger.debug(f"Skipping low quality file: {pose_file.name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load {pose_file}: {e}")
            
            if exercise_data:
                pose_data[exercise_name] = exercise_data
                logger.info(f"Exercise {exercise_name}: {successful_loads}/{len(exercise_files)} files loaded")
            else:
                logger.warning(f"No valid data found for exercise: {exercise_name}")
        
        # Create class mappings
        self.exercise_classes = list(pose_data.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.exercise_classes)}
        
        logger.info(f"Loaded data for {len(self.exercise_classes)} exercise classes")
        
        return pose_data
    
    def create_sliding_windows(self, features, step_size=None):
        """
        Create sliding window sequences from pose features.
        
        Args:
            features (np.array): Pose features (frames, feature_dim)
            step_size (int): Step size for sliding window
            
        Returns:
            list: List of sequences
        """
        if step_size is None:
            step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
        
        sequences = []
        num_frames = len(features)
        
        if num_frames < self.sequence_length:
            # Pad short sequences
            padded_sequence = self.pad_sequence(features, self.sequence_length)
            sequences.append(padded_sequence)
        else:
            # Create overlapping windows
            for start_idx in range(0, num_frames - self.sequence_length + 1, step_size):
                end_idx = start_idx + self.sequence_length
                sequence = features[start_idx:end_idx]
                sequences.append(sequence)
        
        return sequences
    
    def pad_sequence(self, features, target_length):
        """
        Pad sequence to target length by repeating edge frames.
        
        Args:
            features (np.array): Input sequence
            target_length (int): Target sequence length
            
        Returns:
            np.array: Padded sequence
        """
        current_length = len(features)
        
        if current_length >= target_length:
            return features[:target_length]
        
        # Calculate padding needed
        padding_needed = target_length - current_length
        
        # Repeat last frame for padding
        if current_length > 0:
            last_frame = features[-1:].repeat(padding_needed, axis=0)
            padded_sequence = np.concatenate([features, last_frame], axis=0)
        else:
            # Edge case: empty sequence
            padded_sequence = np.zeros((target_length, features.shape[1] if len(features.shape) > 1 else 1))
        
        return padded_sequence
    
    def temporal_augmentation(self, sequence):
        """
        Apply temporal augmentations to a sequence.
        
        Args:
            sequence (np.array): Input sequence (frames, features)
            
        Returns:
            list: List of augmented sequences
        """
        augmented_sequences = []
        
        # Original sequence
        augmented_sequences.append(sequence.copy())
        
        # Speed variations
        if len(sequence) >= self.sequence_length:
            # Faster version (skip frames)
            fast_indices = np.linspace(0, len(sequence)-1, self.sequence_length, dtype=int)
            fast_sequence = sequence[fast_indices]
            augmented_sequences.append(fast_sequence)
            
            # Slower version (interpolate)
            if len(sequence) > self.sequence_length * 0.7:
                slow_indices = np.linspace(0, len(sequence)-1, int(self.sequence_length * 0.8), dtype=int)
                slow_sequence = sequence[slow_indices]
                slow_sequence_padded = self.pad_sequence(slow_sequence, self.sequence_length)
                augmented_sequences.append(slow_sequence_padded)
        
        # Temporal crops (different starting points)
        if len(sequence) > self.sequence_length:
            max_start = len(sequence) - self.sequence_length
            for crop_start in [max_start // 4, max_start // 2, 3 * max_start // 4]:
                if crop_start > 0:
                    cropped_sequence = sequence[crop_start:crop_start + self.sequence_length]
                    if len(cropped_sequence) == self.sequence_length:
                        augmented_sequences.append(cropped_sequence)
        
        # Temporal noise (slight frame shifts)
        noise_sequence = self.add_temporal_noise(sequence)
        augmented_sequences.append(noise_sequence)
        
        return augmented_sequences[:self.augmentation_factor + 1]  # Limit number of augmentations
    
    def spatial_augmentation(self, sequence):
        """
        Apply spatial augmentations to pose landmarks.
        
        Args:
            sequence (np.array): Input sequence (frames, features)
            
        Returns:
            np.array: Spatially augmented sequence
        """
        augmented_sequence = sequence.copy()
        
        # Extract pose landmarks (first 99 features: 33 landmarks × 3)
        if sequence.shape[1] >= 99:
            landmarks = augmented_sequence[:, :99].reshape(-1, 33, 3)
            
            # Horizontal flip
            if random.random() < 0.5:
                landmarks = self.flip_landmarks_horizontal(landmarks)
            
            # Scale variation (±10%)
            if random.random() < 0.3:
                scale_factor = random.uniform(0.9, 1.1)
                landmarks[:, :, :2] *= scale_factor  # Only scale x, y coordinates
            
            # Translation (small shifts)
            if random.random() < 0.3:
                translation = np.random.normal(0, 0.02, 2)  # Small random translation
                landmarks[:, :, :2] += translation
                # Ensure landmarks stay in valid range [0, 1]
                landmarks[:, :, :2] = np.clip(landmarks[:, :, :2], 0, 1)
            
            # Add noise
            if random.random() < 0.4:
                noise = np.random.normal(0, 0.01, landmarks[:, :, :2].shape)
                landmarks[:, :, :2] += noise
                landmarks[:, :, :2] = np.clip(landmarks[:, :, :2], 0, 1)
            
            # Flatten back to original format
            augmented_sequence[:, :99] = landmarks.reshape(-1, 99)
        
        return augmented_sequence
    
    def flip_landmarks_horizontal(self, landmarks):
        """
        Flip pose landmarks horizontally.
        
        Args:
            landmarks (np.array): Landmarks shaped (frames, 33, 3)
            
        Returns:
            np.array: Horizontally flipped landmarks
        """
        flipped = landmarks.copy()
        
        # Flip x coordinates
        flipped[:, :, 0] = 1.0 - flipped[:, :, 0]
        
        # Swap left and right landmarks
        left_right_pairs = [
            (11, 12),  # Shoulders
            (13, 14),  # Elbows
            (15, 16),  # Wrists
            (17, 18),  # Pinkies
            (19, 20),  # Index fingers
            (21, 22),  # Thumbs
            (23, 24),  # Hips
            (25, 26),  # Knees
            (27, 28),  # Ankles
            (29, 30),  # Heels
            (31, 32),  # Foot indices
        ]
        
        for left_idx, right_idx in left_right_pairs:
            flipped[:, [left_idx, right_idx]] = flipped[:, [right_idx, left_idx]]
        
        return flipped
    
    def add_temporal_noise(self, sequence):
        """
        Add temporal noise by slight frame reordering/duplication.
        
        Args:
            sequence (np.array): Input sequence
            
        Returns:
            np.array: Sequence with temporal noise
        """
        noisy_sequence = sequence.copy()
        
        # Randomly duplicate/skip 1-2 frames
        if len(sequence) > self.sequence_length // 2:
            num_modifications = random.randint(0, 2)
            
            for _ in range(num_modifications):
                if random.random() < 0.5 and len(noisy_sequence) < self.sequence_length * 1.2:
                    # Duplicate a random frame
                    dup_idx = random.randint(0, len(noisy_sequence) - 1)
                    noisy_sequence = np.insert(noisy_sequence, dup_idx, noisy_sequence[dup_idx], axis=0)
                elif len(noisy_sequence) > self.sequence_length * 0.8:
                    # Remove a random frame
                    remove_idx = random.randint(0, len(noisy_sequence) - 1)
                    noisy_sequence = np.delete(noisy_sequence, remove_idx, axis=0)
        
        # Ensure final sequence is correct length
        if len(noisy_sequence) != self.sequence_length:
            noisy_sequence = self.pad_sequence(noisy_sequence, self.sequence_length)
            if len(noisy_sequence) > self.sequence_length:
                noisy_sequence = noisy_sequence[:self.sequence_length]
        
        return noisy_sequence
    
    def prepare_sequences(self, pose_data, apply_augmentation=True):
        """
        Prepare training sequences from pose data.
        
        Args:
            pose_data (dict): Loaded pose data from load_pose_data
            apply_augmentation (bool): Whether to apply data augmentation
            
        Returns:
            tuple: (sequences, labels, metadata)
        """
        all_sequences = []
        all_labels = []
        sequence_metadata = []
        
        logger.info("Preparing training sequences...")
        
        for exercise_name, exercise_data in pose_data.items():
            exercise_label = self.class_to_idx[exercise_name]
            exercise_sequences = 0
            
            logger.info(f"Processing {len(exercise_data)} videos for exercise: {exercise_name}")
            
            for video_data in tqdm(exercise_data, desc=f"Creating sequences for {exercise_name}"):
                features = video_data['features']
                
                # Create sliding window sequences
                base_sequences = self.create_sliding_windows(features)
                
                for seq_idx, base_sequence in enumerate(base_sequences):
                    # Add original sequence
                    all_sequences.append(base_sequence)
                    all_labels.append(exercise_label)
                    sequence_metadata.append({
                        'exercise': exercise_name,
                        'source_file': video_data['filename'],
                        'sequence_index': seq_idx,
                        'augmentation_type': 'original',
                        'detection_info': video_data['detection_info']
                    })
                    exercise_sequences += 1
                    
                    # Apply augmentations if enabled
                    if apply_augmentation:
                        augmented_sequences = self.temporal_augmentation(base_sequence)
                        
                        for aug_idx, aug_sequence in enumerate(augmented_sequences[1:]):  # Skip original
                            # Apply spatial augmentation
                            spatially_augmented = self.spatial_augmentation(aug_sequence)
                            
                            all_sequences.append(spatially_augmented)
                            all_labels.append(exercise_label)
                            sequence_metadata.append({
                                'exercise': exercise_name,
                                'source_file': video_data['filename'],
                                'sequence_index': seq_idx,
                                'augmentation_type': f'temporal_spatial_{aug_idx}',
                                'detection_info': video_data['detection_info']
                            })
                            exercise_sequences += 1
            
            logger.info(f"Exercise {exercise_name}: {exercise_sequences} total sequences created")
        
        logger.info(f"Total sequences created: {len(all_sequences)}")
        
        return np.array(all_sequences), np.array(all_labels), sequence_metadata
    
    def create_dataset_splits(self, sequences, labels, metadata, test_size=0.15, val_size=0.15, random_state=42):
        """
        Create stratified train/validation/test splits.
        
        Args:
            sequences (np.array): All sequences
            labels (np.array): All labels
            metadata (list): Sequence metadata
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set
            random_state (int): Random seed
            
        Returns:
            tuple: Split data and metadata
        """
        logger.info("Creating dataset splits...")
        
        # Group sequences by source file to avoid data leakage
        file_groups = defaultdict(list)
        for idx, meta in enumerate(metadata):
            key = f"{meta['exercise']}_{meta['source_file']}"
            file_groups[key].append(idx)
        
        # Create file-level splits first
        file_keys = list(file_groups.keys())
        file_labels = [metadata[file_groups[key][0]]['exercise'] for key in file_keys]
        
        # Convert exercise names to indices for stratification
        file_label_indices = [self.class_to_idx[label] for label in file_labels]
        
        # Split files, not individual sequences
        train_files, temp_files, train_file_labels, temp_file_labels = train_test_split(
            file_keys, file_label_indices, test_size=test_size + val_size,
            stratify=file_label_indices, random_state=random_state
        )
        
        val_ratio = val_size / (test_size + val_size)
        val_files, test_files, val_file_labels, test_file_labels = train_test_split(
            temp_files, temp_file_labels, test_size=1-val_ratio,
            stratify=temp_file_labels, random_state=random_state
        )
        
        # Map back to sequence indices
        train_indices = []
        val_indices = []
        test_indices = []
        
        for file_key in train_files:
            train_indices.extend(file_groups[file_key])
        for file_key in val_files:
            val_indices.extend(file_groups[file_key])
        for file_key in test_files:
            test_indices.extend(file_groups[file_key])
        
        # Create splits
        X_train = sequences[train_indices]
        y_train = labels[train_indices]
        train_metadata = [metadata[i] for i in train_indices]
        
        X_val = sequences[val_indices]
        y_val = labels[val_indices]
        val_metadata = [metadata[i] for i in val_indices]
        
        X_test = sequences[test_indices]
        y_test = labels[test_indices]
        test_metadata = [metadata[i] for i in test_indices]
        
        logger.info(f"Dataset splits created:")
        logger.info(f"  Training: {len(X_train)} sequences")
        logger.info(f"  Validation: {len(X_val)} sequences")
        logger.info(f"  Test: {len(X_test)} sequences")
        
        # Check class distribution
        for split_name, split_labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            class_counts = np.bincount(split_labels)
            class_dist = class_counts / len(split_labels) * 100
            logger.info(f"  {split_name} class distribution: {class_dist}")
        
        return (X_train, y_train, train_metadata), (X_val, y_val, val_metadata), (X_test, y_test, test_metadata)
    
    def normalize_features(self, X_train, X_val, X_test):
        """
        Normalize features using training set statistics.
        
        Args:
            X_train, X_val, X_test (np.array): Dataset splits
            
        Returns:
            tuple: Normalized datasets and scaler
        """
        logger.info("Normalizing features...")
        
        # Reshape for normalization (flatten sequence dimension)
        original_shape = X_train.shape
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        
        # Fit scaler on training data only
        scaler = StandardScaler()
        X_train_norm_flat = scaler.fit_transform(X_train_flat)
        X_val_norm_flat = scaler.transform(X_val_flat)
        X_test_norm_flat = scaler.transform(X_test_flat)
        
        # Reshape back to sequence format
        X_train_norm = X_train_norm_flat.reshape(original_shape)
        X_val_norm = X_val_norm_flat.reshape(X_val.shape)
        X_test_norm = X_test_norm_flat.reshape(X_test.shape)
        
        logger.info("Feature normalization completed")
        
        return X_train_norm, X_val_norm, X_test_norm, scaler
    
    def save_dataset(self, train_data, val_data, test_data, scaler, output_folder):
        """
        Save prepared dataset to files.
        
        Args:
            train_data, val_data, test_data (tuple): Dataset splits with metadata
            scaler: Fitted StandardScaler
            output_folder (str): Output directory
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving dataset to: {output_folder}")
        
        # Unpack data
        X_train, y_train, train_metadata = train_data
        X_val, y_val, val_metadata = val_data
        X_test, y_test, test_metadata = test_data
        
        # Save arrays
        np.save(output_path / "X_train.npy", X_train)
        np.save(output_path / "y_train.npy", y_train)
        np.save(output_path / "X_val.npy", X_val)
        np.save(output_path / "y_val.npy", y_val)
        np.save(output_path / "X_test.npy", X_test)
        np.save(output_path / "y_test.npy", y_test)
        
        # Save scaler
        with open(output_path / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save metadata and class information
        dataset_info = {
            'exercise_classes': self.exercise_classes,
            'class_to_idx': self.class_to_idx,
            'sequence_length': self.sequence_length,
            'num_features': X_train.shape[-1],
            'dataset_shapes': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            },
            'train_metadata': train_metadata,
            'val_metadata': val_metadata,
            'test_metadata': test_metadata
        }
        
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info("Dataset saved successfully")
        
        return dataset_info

def run_sequence_preparation(pose_features_folder, output_folder, apply_augmentation=True):
    """
    Main function to run sequence preparation pipeline.
    
    Args:
        pose_features_folder (str): Path to pose features from Step 3
        output_folder (str): Path to save prepared dataset
        apply_augmentation (bool): Whether to apply data augmentation
        
    Returns:
        dict: Processing results
    """
    preparator = SequencePreparator()
    
    # Load pose data
    pose_data = preparator.load_pose_data(pose_features_folder)
    
    if not pose_data:
        logger.error("No valid pose data found")
        return None
    
    # Prepare sequences
    sequences, labels, metadata = preparator.prepare_sequences(pose_data, apply_augmentation)
    
    # Create dataset splits
    train_data, val_data, test_data = preparator.create_dataset_splits(sequences, labels, metadata)
    
    # Normalize features
    X_train_norm, X_val_norm, X_test_norm, scaler = preparator.normalize_features(
        train_data[0], val_data[0], test_data[0]
    )
    
    # Update data with normalized features
    train_data = (X_train_norm, train_data[1], train_data[2])
    val_data = (X_val_norm, val_data[1], val_data[2])
    test_data = (X_test_norm, test_data[1], test_data[2])
    
    # Save dataset
    dataset_info = preparator.save_dataset(train_data, val_data, test_data, scaler, output_folder)
    
    # Generate final report
    print("\n" + "="*80)
    print("SEQUENCE PREPARATION COMPLETE")
    print("="*80)
    print(f"Final dataset shapes:")
    print(f"  Training: {dataset_info['dataset_shapes']['train']}")
    print(f"  Validation: {dataset_info['dataset_shapes']['val']}")
    print(f"  Test: {dataset_info['dataset_shapes']['test']}")
    print(f"Exercise classes: {dataset_info['exercise_classes']}")
    print(f"Features per frame: {dataset_info['num_features']}")
    print(f"Sequence length: {dataset_info['sequence_length']} frames")
    print(f"Dataset saved to: {output_folder}")
    
    return dataset_info

if __name__ == "__main__":
    # Example usage
    pose_features_folder = "/path/to/pose/features"
    output_folder = "/path/to/final/dataset"
    
    try:
        dataset_info = run_sequence_preparation(
            pose_features_folder=pose_features_folder,
            output_folder=output_folder,
            apply_augmentation=True
        )
        
        if dataset_info:
            print(f"\nDataset preparation complete!")
            print(f"Ready for CNN+LSTM model training")
        else:
            print("Dataset preparation failed")
            
    except Exception as e:
        logger.error(f"Error during sequence preparation: {e}")