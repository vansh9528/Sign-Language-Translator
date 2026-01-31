"""
Real-time Video Gesture Recognition

Uses temporal/motion features for gesture prediction.

Usage:
    python live_realtime_video.py
"""

import cv2
import pickle
import numpy as np
from collections import deque

from config import (
    WINDOW_NAME,
    FONT_SCALE,
    FONT_COLOR,
    FONT_THICKNESS,
    CAMERA_BACKEND,
    STABLE_FRAMES,
    CONFIDENCE_THRESHOLD
)
from utils import (
    initialize_hand_detector,
    initialize_camera,
    extract_hand_landmarks,
    draw_landmarks_on_frame,
    add_text_to_frame
)


def load_model(model_path):
    """Load trained video gesture model."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Run: python train_video_model.py")
        exit(1)


def extract_temporal_features(landmarks_buffer, num_frames=5):
    """
    Extract temporal features from landmark buffer.
    
    Args:
        landmarks_buffer: Deque of landmark vectors
        num_frames: Number of frames to use
    
    Returns:
        list: Temporal feature vector
    """
    if len(landmarks_buffer) < num_frames:
        # Pad with zeros
        landmarks_list = list(landmarks_buffer) + [
            [0] * 63 for _ in range(num_frames - len(landmarks_buffer))
        ]
    else:
        landmarks_list = list(landmarks_buffer)[-num_frames:]
    
    temporal_features = []
    
    # 1. Stack current landmarks
    for frame in landmarks_list:
        temporal_features.extend(frame)
    
    # 2. Compute velocity
    for i in range(1, len(landmarks_list)):
        velocity = [
            landmarks_list[i][j] - landmarks_list[i-1][j]
            for j in range(len(landmarks_list[i]))
        ]
        temporal_features.extend(velocity)
    
    # 3. Compute acceleration
    if len(landmarks_list) > 2:
        for i in range(2, len(landmarks_list)):
            accel = [
                (landmarks_list[i][j] - landmarks_list[i-1][j]) -
                (landmarks_list[i-1][j] - landmarks_list[i-2][j])
                for j in range(len(landmarks_list[i]))
            ]
            temporal_features.extend(accel)
    
    return temporal_features


def run_video_prediction(model, sequence_length=5):
    """
    Run real-time video gesture prediction.
    
    Args:
        model: Trained video gesture model
        sequence_length: Frames to buffer for temporal features
    """
    hands, mp_hands = initialize_hand_detector()
    cap = initialize_camera(CAMERA_BACKEND)
    
    landmarks_buffer = deque(maxlen=sequence_length)
    last_prediction = ""
    stable_count = 0
    
    print(f"\n{'='*60}")
    print("VIDEO GESTURE RECOGNITION - REAL-TIME")
    print(f"{'='*60}")
    print("Show your hand and perform gestures")
    print("Press ESC to exit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        display_text = "..."
        confidence_text = ""
        buffer_status = ""
        
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            frame = draw_landmarks_on_frame(frame, hand, mp_hands)
            
            # Extract static landmarks
            features = extract_hand_landmarks(hand)
            landmarks_buffer.append(features)
            
            # Show buffer status
            buffer_status = f"Buffer: {len(landmarks_buffer)}/{sequence_length}"
            
            # Make prediction if buffer is full
            if len(landmarks_buffer) == sequence_length:
                temporal_features = extract_temporal_features(landmarks_buffer, sequence_length)
                
                probs = model.predict_proba([temporal_features])[0]
                pred = model.classes_[np.argmax(probs)]
                confidence = np.max(probs)
                
                # Stability check
                if pred == last_prediction:
                    stable_count += 1
                else:
                    stable_count = 0
                
                # Display if stable and confident
                if stable_count >= STABLE_FRAMES and confidence > CONFIDENCE_THRESHOLD:
                    display_text = pred
                
                last_prediction = pred
                confidence_text = f"Conf: {confidence:.2f} | Stable: {stable_count}/{STABLE_FRAMES}"
                
                # Print debug
                print(f"{pred:15} | {confidence:.2f} | {stable_count}/{STABLE_FRAMES}")
        
        # Draw text on frame
        frame = add_text_to_frame(
            frame,
            f"Gesture: {display_text}",
            position=(30, 40),
            font_scale=FONT_SCALE,
            color=FONT_COLOR,
            thickness=FONT_THICKNESS
        )
        
        if buffer_status:
            frame = add_text_to_frame(
                frame,
                buffer_status,
                position=(30, 80),
                font_scale=0.7,
                color=(255, 255, 0),
                thickness=1
            )
        
        if confidence_text:
            frame = add_text_to_frame(
                frame,
                confidence_text,
                position=(30, 120),
                font_scale=0.7,
                color=(200, 255, 200),
                thickness=1
            )
        
        cv2.imshow(WINDOW_NAME + " (Video)", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Video gesture recognition ended")


if __name__ == "__main__":
    model = load_model("gesture_model_video.pkl")
    run_video_prediction(model, sequence_length=5)
