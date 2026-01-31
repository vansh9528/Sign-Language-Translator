"""
Train Sign Language Model on Video Sequences

This script trains an SVM model using temporal features from video sequences.
Captures not just hand positions, but also motion and dynamics.

Usage:
    python train_video_model.py
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from config import (
    DATASET_FOLDER,
    TEST_SIZE,
    RANDOM_STATE,
    SVM_KERNEL,
    SVM_C,
    SVM_GAMMA,
    SVM_PROBABILITY
)


def load_video_dataset(dataset_folder):
    """
    Load video-based gesture data (with temporal features).
    
    IMPORTANT: Only loads _VIDEO.csv files with temporal features.
    Does NOT mix with static features (different dimensions).
    
    Args:
        dataset_folder (str): Path to dataset folder
    
    Returns:
        tuple: (X, y) features and labels
    """
    import numpy as np
    
    X = []
    y = []
    
    if not os.path.exists(dataset_folder):
        print(f"❌ Dataset folder not found: {dataset_folder}")
        exit(1)
    
    files = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]
    
    if not files:
        print(f"❌ No CSV files found in {dataset_folder}")
        exit(1)
    
    print(f"\nLoading video gesture data from {dataset_folder}/")
    print(f"{'='*60}")
    
    # Load ONLY video data (temporal features - 504D)
    video_files = [f for f in files if "_VIDEO" in f]
    
    if not video_files:
        print(f"⚠️  WARNING: No *_VIDEO.csv files found!")
        print(f"Please collect video gesture data first:")
        print(f"  python collect_video_data.py")
        exit(1)
    
    # Load video data (temporal features)
    for file in sorted(video_files):
        label = file.replace("_VIDEO.csv", "")
        file_path = os.path.join(dataset_folder, file)
        
        try:
            data = pd.read_csv(file_path, header=None)
            # Convert to numpy array to ensure consistency
            samples = data.values
            
            # Validate feature dimension
            if len(samples) > 0:
                feature_dim = len(samples[0])
                print(f"✓ VIDEO: {label:15} - {len(samples):4} samples (dim: {feature_dim})")
                
                for sample in samples:
                    # Ensure each sample is a flat 1D array
                    if isinstance(sample, np.ndarray):
                        X.append(sample.flatten())
                    else:
                        X.append(np.array(sample).flatten())
                    y.append(label)
            else:
                print(f"⚠️  EMPTY: {label:15} - {file}")
        except Exception as e:
            print(f"✗ Error loading {file}: {e}")
    
    # Convert to numpy arrays
    if X:
        X = np.array(X, dtype=np.float32)
    
    return X, y


def train_video_model(X_train, y_train):
    """
    Train SVM model on video/temporal features.
    
    Args:
        X_train: Training features (numpy array)
        y_train: Training labels
    
    Returns:
        Pipeline: Trained model
    """
    print(f"\n{'='*60}")
    print("Training Video-Based SVM Model")
    print(f"{'='*60}")
    
    # Ensure X_train is numpy array
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train, dtype=np.float32)
    
    print(f"Model configuration:")
    print(f"  - Kernel: {SVM_KERNEL}")
    print(f"  - C: {SVM_C}")
    print(f"  - Gamma: {SVM_GAMMA}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Features per sample: {X_train.shape[1] if X_train.ndim > 1 else 0}")
    print(f"  - Array shape: {X_train.shape}")
    
    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA,
            probability=SVM_PROBABILITY,
            class_weight="balanced"
        )
    )
    
    model.fit(X_train, y_train)
    
    print(f"✓ Model trained successfully")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print("Model Evaluation")
    print(f"{'='*60}")
    print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))


def save_model(model, model_path):
    """Save trained model."""
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\n{'='*60}")
    print(f"✓ Model saved to {model_path}")
    print(f"{'='*60}")


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("VIDEO GESTURE RECOGNITION - MODEL TRAINING")
    print("="*60)
    
    # Load data
    X, y = load_video_dataset(DATASET_FOLDER)
    
    if len(X) == 0:
        print("❌ No training data loaded!")
        exit(1)
    
    X = np.array(X, dtype=np.float32) if not isinstance(X, np.ndarray) else X
    
    print(f"\n📊 Dataset Summary:")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Total gestures: {len(set(y))}")
    print(f"  - Features per sample: {X.shape[1] if X.ndim > 1 else 0}")
    print(f"  - Unique gestures: {sorted(set(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n📈 Train/Test Split:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    
    # Train model
    model = train_video_model(X_train, y_train)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Save
    model_path = "gesture_model_video.pkl"
    save_model(model, model_path)
    
    print(f"\n✓ Video gesture training complete!")
    print(f"Use with: python live_realtime_video.py")


if __name__ == "__main__":
    main()
