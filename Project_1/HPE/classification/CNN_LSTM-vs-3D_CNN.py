import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
import pickle
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExerciseRecognitionModel:
    """
    Exercise recognition model with both CNN+LSTM and 3D CNN architectures.
    """
    
    def __init__(self, num_classes, sequence_length, num_features, model_type='3dcnn'):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def build_cnn_lstm_model(self, lstm_units=64, cnn_filters=[32, 64], dropout_rate=0.3):
        """
        Build CNN+LSTM model as described in the research paper.
        
        Args:
            lstm_units (int): Number of LSTM units
            cnn_filters (list): CNN filter sizes for feature extraction
            dropout_rate (float): Dropout rate for regularization
        """
        logger.info("Building CNN+LSTM model...")
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.num_features), name='pose_input')
        
        # Reshape for CNN processing (treat features as 1D spatial dimension)
        # Shape: (batch, sequence_length, num_features, 1)
        x = layers.Reshape((self.sequence_length, self.num_features, 1))(inputs)
        
        # CNN Feature Extraction
        # Process each frame independently with 1D convolutions across features
        x = layers.TimeDistributed(
            layers.Conv1D(filters=cnn_filters[0], kernel_size=3, activation='relu', padding='same'),
            name='cnn_layer_1'
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(x)
        x = layers.TimeDistributed(layers.Dropout(dropout_rate/2))(x)
        
        x = layers.TimeDistributed(
            layers.Conv1D(filters=cnn_filters[1], kernel_size=3, activation='relu', padding='same'),
            name='cnn_layer_2'
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(x)
        x = layers.TimeDistributed(layers.Dropout(dropout_rate/2))(x)
        
        # Flatten features for each timestep
        x = layers.TimeDistributed(layers.Flatten())(x)
        
        # Dense layer to compress features
        x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
        x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)
        
        # LSTM layers for temporal modeling
        x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)(x)
        x = layers.LSTM(lstm_units//2, return_sequences=False, dropout=dropout_rate)(x)
        
        # Classification head
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='exercise_output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_ExerciseRecognition')
        
        logger.info(f"CNN+LSTM model built with {model.count_params()} parameters")
        return model
    
    def build_3dcnn_model(self, filters=[16, 32, 64], dropout_rate=0.3):
        """
        Build 3D CNN model for exercise recognition.
        
        Args:
            filters (list): 3D CNN filter sizes
            dropout_rate (float): Dropout rate for regularization
        """
        logger.info("Building 3D CNN model...")
        
        # Reshape input for 3D CNN: (batch, time, height, width, channels)
        # We'll treat features as spatial dimensions
        height = int(np.sqrt(self.num_features)) if int(np.sqrt(self.num_features))**2 == self.num_features else 10
        width = self.num_features // height
        
        inputs = layers.Input(shape=(self.sequence_length, self.num_features), name='pose_input')
        
        # Reshape to pseudo-spatial format for 3D CNN
        x = layers.Reshape((self.sequence_length, height, width, 1))(inputs)
        
        # 3D CNN blocks
        # Block 1
        x = layers.Conv3D(filters=filters[0], kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
        x = layers.Dropout(dropout_rate/3)(x)
        
        # Block 2
        x = layers.Conv3D(filters=filters[1], kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = layers.Dropout(dropout_rate/2)(x)
        
        # Block 3
        x = layers.Conv3D(filters=filters[2], kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='exercise_output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='3DCNN_ExerciseRecognition')
        
        logger.info(f"3D CNN model built with {model.count_params()} parameters")
        return model
    
    def build_simple_3dcnn_model(self, filters=[32, 64], dropout_rate=0.3):
        """
        Build simpler 3D CNN that works directly with feature vectors.
        """
        logger.info("Building Simple 3D CNN model...")
        
        inputs = layers.Input(shape=(self.sequence_length, self.num_features), name='pose_input')
        
        # Reshape to (batch, time, features, 1, 1) for 3D conv
        x = layers.Reshape((self.sequence_length, self.num_features, 1, 1))(inputs)
        
        # 3D CNN layers
        x = layers.Conv3D(filters=filters[0], kernel_size=(3, 5, 1), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(pool_size=(1, 2, 1))(x)
        x = layers.Dropout(dropout_rate/2)(x)
        
        x = layers.Conv3D(filters=filters[1], kernel_size=(3, 5, 1), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='Simple3DCNN_ExerciseRecognition')
        
        logger.info(f"Simple 3D CNN model built with {model.count_params()} parameters")
        return model
    
    def build_model(self, **kwargs):
        """
        Build the specified model architecture.
        """
        if self.model_type == 'cnn_lstm':
            self.model = self.build_cnn_lstm_model(**kwargs)
        elif self.model_type == '3dcnn':
            self.model = self.build_simple_3dcnn_model(**kwargs)
        elif self.model_type == 'complex_3dcnn':
            self.model = self.build_3dcnn_model(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """
        Compile the model with appropriate loss and metrics.
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        logger.info(f"Model compiled with {optimizer} optimizer (lr={learning_rate})")
    
    def create_callbacks(self, model_dir, patience=15):
        """
        Create training callbacks for monitoring and saving.
        """
        callbacks_list = [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, f'best_{self.model_type}_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.CSVLogger(
                os.path.join(model_dir, f'{self.model_type}_training_log.csv')
            )
        ]
        
        return callbacks_list
    
    def train(self, X_train, y_train, X_val, y_val, model_dir, epochs=100, batch_size=32):
        """
        Train the model with proper monitoring and callbacks.
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        # Create model directory
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model architecture
        with open(os.path.join(model_dir, f'{self.model_type}_architecture.json'), 'w') as f:
            f.write(self.model.to_json())
        
        # Create callbacks
        callback_list = self.create_callbacks(model_dir)
        
        logger.info(f"Starting {self.model_type} training...")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        start_time = time.time()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save training history
        with open(os.path.join(model_dir, f'{self.model_type}_history.pkl'), 'wb') as f:
            pickle.dump(self.history.history, f)
        
        return self.history
    
    def evaluate(self, X_test, y_test, class_names):
        """
        Evaluate model performance and generate detailed reports.
        """
        if self.model is None:
            raise ValueError("Model must be built before evaluation")
        
        logger.info("Evaluating model performance...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy, test_top2_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top2_accuracy': test_top2_accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Top-2 Accuracy: {test_top2_accuracy:.4f}")
        
        return results
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history curves.
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_type.upper()} Training History', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-2 Accuracy
        if 'top_2_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_2_accuracy'], label='Training Top-2 Accuracy')
            axes[1, 0].plot(self.history.history['val_top_2_accuracy'], label='Validation Top-2 Accuracy')
            axes[1, 0].set_title('Top-2 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-2 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names, save_path=None):
        """
        Plot confusion matrix.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{self.model_type.upper()} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()

def load_dataset(dataset_folder):
    """
    Load the prepared dataset from Step 4.
    """
    dataset_path = Path(dataset_folder)
    
    # Load arrays
    X_train = np.load(dataset_path / "X_train.npy")
    y_train = np.load(dataset_path / "y_train.npy")
    X_val = np.load(dataset_path / "X_val.npy")
    y_val = np.load(dataset_path / "y_val.npy")
    X_test = np.load(dataset_path / "X_test.npy")
    y_test = np.load(dataset_path / "y_test.npy")
    
    # Load metadata
    with open(dataset_path / "dataset_info.json", 'r') as f:
        dataset_info = json.load(f)
    
    # Load scaler
    with open(dataset_path / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    logger.info(f"Dataset loaded successfully")
    logger.info(f"Training shape: {X_train.shape}")
    logger.info(f"Classes: {dataset_info['exercise_classes']}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), dataset_info, scaler

def compare_models(dataset_folder, output_folder, epochs=100, batch_size=32):
    """
    Train and compare both CNN+LSTM and 3D CNN models.
    """
    # Load dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test), dataset_info, scaler = load_dataset(dataset_folder)
    
    num_classes = len(dataset_info['exercise_classes'])
    sequence_length = dataset_info['sequence_length']
    num_features = dataset_info['num_features']
    class_names = dataset_info['exercise_classes']
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Train both models
    for model_type in ['cnn_lstm', '3dcnn']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type.upper()} Model")
        logger.info(f"{'='*60}")
        
        # Create model directory
        model_dir = output_path / model_type
        model_dir.mkdir(exist_ok=True)
        
        # Initialize and build model
        model = ExerciseRecognitionModel(
            num_classes=num_classes,
            sequence_length=sequence_length,
            num_features=num_features,
            model_type=model_type
        )
        
        model.build_model()
        model.compile_model(learning_rate=0.001)
        
        # Print model summary
        print(f"\n{model_type.upper()} Model Summary:")
        model.model.summary()
        
        # Train model
        history = model.train(
            X_train, y_train, X_val, y_val,
            model_dir=str(model_dir),
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate model
        evaluation_results = model.evaluate(X_test, y_test, class_names)
        
        # Plot results
        model.plot_training_history(save_path=str(model_dir / f'{model_type}_training_history.png'))
        model.plot_confusion_matrix(
            evaluation_results['confusion_matrix'],
            class_names,
            save_path=str(model_dir / f'{model_type}_confusion_matrix.png')
        )
        
        # Save results
        results[model_type] = {
            'model_params': model.model.count_params(),
            'test_accuracy': evaluation_results['test_accuracy'],
            'test_top2_accuracy': evaluation_results['test_top2_accuracy'],
            'test_loss': evaluation_results['test_loss'],
            'classification_report': evaluation_results['classification_report']
        }
        
        # Save detailed results
        with open(model_dir / f'{model_type}_results.json', 'w') as f:
            json.dump(results[model_type], f, indent=2)
    
    # Compare results
    print(f"\n{'='*60}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*60}")
    
    comparison = []
    for model_type, result in results.items():
        comparison.append({
            'Model': model_type.upper(),
            'Parameters': f"{result['model_params']:,}",
            'Test Accuracy': f"{result['test_accuracy']:.4f}",
            'Top-2 Accuracy': f"{result['test_top2_accuracy']:.4f}",
            'Test Loss': f"{result['test_loss']:.4f}"
        })
    
    import pandas as pd
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(output_path / 'model_comparison.csv', index=False)
    
    # Determine best model
    best_model = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    print(f"\nBest performing model: {best_model.upper()}")
    print(f"Best accuracy: {results[best_model]['test_accuracy']:.4f}")
    
    return results, best_model

if __name__ == "__main__":
    # Configuration
    dataset_folder = "/path/to/final/dataset"  # From Step 4
    output_folder = "/path/to/model/results"
    
    # Training parameters
    epochs = 100
    batch_size = 32
    
    try:
        # Run comparison
        results, best_model = compare_models(
            dataset_folder=dataset_folder,
            output_folder=output_folder,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print(f"\nTraining complete! Results saved to: {output_folder}")
        print(f"Best model: {best_model}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise 