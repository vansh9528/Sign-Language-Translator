"""
Train Sign Language Gesture Recognition Model

This script trains an SVM model on hand landmark data collected from multiple gestures.
It uses scikit-learn's SVC with a StandardScaler pipeline for preprocessing.

Usage:
    python train_landmark_model.py
"""

import os
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
    MODEL_PATH,
    SVM_KERNEL,
    SVM_C,
    SVM_GAMMA,
    SVM_PROBABILITY
)


def load_dataset(dataset_folder):
    """
    Load hand landmark data from CSV files.
    
    Args:
        dataset_folder (str): Path to folder containing gesture CSV files
    
    Returns:
        tuple: (X, y) features and labels
    """
    X = []
    y = []
    
    if not os.path.exists(dataset_folder):
        print(f"❌ Dataset folder not found: {dataset_folder}")
        print(f"Please collect data first using: python collect_data.py")
        exit(1)
    
    files = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]
    
    if not files:
        print(f"❌ No CSV files found in {dataset_folder}")
        print(f"Please collect data first using: python collect_data.py")
        exit(1)
    
    print(f"\nLoading data from {dataset_folder}/")
    print(f"{'='*50}")
    
    for file in sorted(files):
        label = file.replace(".csv", "")
        file_path = os.path.join(dataset_folder, file)
        
        data = pd.read_csv(file_path, header=None)
        
        X.extend(data.values)
        y.extend([label] * len(data))
        
        print(f"✓ {label:15} - {len(data):4} samples")
    
    return X, y


def train_model(X_train, y_train):
    """
    Train SVM model with preprocessing pipeline.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Pipeline: Trained model pipeline
    """
    print(f"\n{'='*50}")
    print("Training SVM Model")
    print(f"{'='*50}")
    
    # Create pipeline: StandardScaler -> SVC
    # StandardScaler is crucial for SVM performance with landmarks
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
    
    print(f"Model configuration:")
    print(f"  - Kernel: {SVM_KERNEL}")
    print(f"  - C: {SVM_C}")
    print(f"  - Gamma: {SVM_GAMMA}")
    print(f"  - Training samples: {len(X_train)}")
    
    model.fit(X_train, y_train)
    
    print(f"✓ Model trained successfully")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print("Model Evaluation")
    print(f"{'='*50}")
    print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))


def save_model(model, model_path):
    """
    Save trained model to pickle file.
    
    Args:
        model: Trained model
        model_path (str): Path to save model
    """
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\n{'='*50}")
    print(f"✓ Model saved to {model_path}")
    print(f"{'='*50}")


def main():
    """Main training pipeline."""
    print("\n" + "="*50)
    print("SIGN LANGUAGE TRANSLATOR - MODEL TRAINING")
    print("="*50)
    
    # Load data
    X, y = load_dataset(DATASET_FOLDER)
    
    print(f"\n📊 Dataset Summary:")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Total gestures: {len(set(y))}")
    print(f"  - Features per sample: {len(X[0])}")
    
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
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, MODEL_PATH)
    
    print(f"\n✓ Training complete! Ready for real-time prediction.")


if __name__ == "__main__":
    main()
