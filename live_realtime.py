"""
Real-time Sign Language Gesture Recognition

This script runs real-time prediction using a trained SVM model on hand landmarks.

Usage:
    python live_realtime.py
"""

import cv2
import pickle
import numpy as np
from config import (
    MODEL_PATH,
    STABLE_FRAMES,
    CONFIDENCE_THRESHOLD,
    DISPLAY_ON_UNSTABLE,
    CAMERA_BACKEND,
    WINDOW_NAME,
    FONT_SCALE,
    FONT_COLOR,
    FONT_THICKNESS
)
from utils import (
    initialize_hand_detector,
    initialize_camera,
    extract_hand_landmarks,
    draw_landmarks_on_frame,
    add_text_to_frame
)


def load_model(model_path):
    """Load trained SVM model from pickle file."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using: python train_landmark_model.py")
        exit(1)


def run_real_time_prediction(model):
    """
    Run real-time gesture prediction using webcam feed.
    
    Args:
        model: Trained SVM model
    """
    # Initialize detector and camera
    hands, mp_hands = initialize_hand_detector()
    cap = initialize_camera(CAMERA_BACKEND)
    
    last_prediction = ""
    stable_count = 0
    
    print(f"\n{'='*50}")
    print("REAL-TIME GESTURE RECOGNITION")
    print(f"{'='*50}")
    print("Show your hand to the camera")
    print("Press ESC to exit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        display_text = DISPLAY_ON_UNSTABLE
        confidence_text = ""
        
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            
            # Draw landmarks
            frame = draw_landmarks_on_frame(frame, hand, mp_hands)
            
            # Extract normalized features
            features = extract_hand_landmarks(hand)
            
            # Make prediction
            probs = model.predict_proba([features])[0]
            pred = model.classes_[np.argmax(probs)]
            confidence = np.max(probs)
            
            # Stability check
            if pred == last_prediction:
                stable_count += 1
            else:
                stable_count = 0
            
            # Display gesture if stable and confident
            if stable_count >= STABLE_FRAMES and confidence > CONFIDENCE_THRESHOLD:
                display_text = pred
            
            last_prediction = pred
            confidence_text = f"Confidence: {confidence:.2f}"
            
            # Print debug info
            print(f"{pred:15} | Confidence: {confidence:.2f} | Stable: {stable_count}/{STABLE_FRAMES}")
        
        # Draw text on frame
        frame = add_text_to_frame(
            frame,
            f"Gesture: {display_text}",
            position=(30, 40),
            font_scale=FONT_SCALE,
            color=FONT_COLOR,
            thickness=FONT_THICKNESS
        )
        
        if confidence_text:
            frame = add_text_to_frame(
                frame,
                confidence_text,
                position=(30, 80),
                font_scale=0.7,
                color=(255, 255, 0),
                thickness=1
            )
        
        cv2.imshow(WINDOW_NAME, frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Real-time prediction ended")


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    run_real_time_prediction(model)
