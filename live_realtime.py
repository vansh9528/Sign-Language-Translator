"""
Real-time Sign Language Gesture Recognition

This script runs real-time prediction using a trained SVM model on hand landmarks.

Usage:
    python live_realtime.py
"""

import cv2
import pickle
import numpy as np
import time
from collections import deque, Counter
from config import (
    MODEL_PATH,
    STABLE_FRAMES,
    CONFIDENCE_THRESHOLD,
    DISPLAY_ON_UNSTABLE,
    CAMERA_BACKEND,
    USE_ESP32,
    ESP32_STREAM_URL,
    FLIP_FRAME,
    WINDOW_NAME,
    FONT_SCALE,
    FONT_COLOR,
    FONT_THICKNESS
)
from utils_fixed import (
    initialize_hand_detector,
    initialize_camera,
    extract_hand_landmarks,
    draw_landmarks_on_frame,
    add_text_to_frame,
    create_mjpeg_stream,
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
    if USE_ESP32:
        print(f"Using ESP32-CAM stream (URL from config)")
    
    last_prediction = ""
    stable_count = 0
    # deque used for short-window majority smoothing to avoid sticky/incorrect labels
    prediction_buffer = deque(maxlen=STABLE_FRAMES)
    
    print(f"\n{'='*50}")
    print("REAL-TIME GESTURE RECOGNITION")
    print(f"{'='*50}")
    print("Show your hand to the camera")
    print("Press ESC to exit\n")
    
    read_failures = 0
    MAX_CONSECUTIVE_READ_FAILURES = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            read_failures += 1
            print(f"❌ Frame not received from capture (consecutive failures: {read_failures})")
            # If using ESP32, try reconnection first; after a few failures switch to MJPEG fallback
            if USE_ESP32:
                if read_failures < MAX_CONSECUTIVE_READ_FAILURES:
                    print("Attempting to reconnect to ESP32 stream...")
                    try:
                        cap = initialize_camera(CAMERA_BACKEND)
                        continue
                    except Exception as e:
                        print(f"Reconnect failed: {e}")
                        time.sleep(0.5)
                        continue
                else:
                    print("Switching to MJPEG fallback after repeated read failures")
                    try:
                        cap = create_mjpeg_stream(ESP32_STREAM_URL)
                        read_failures = 0
                        print("MJPEG fallback active")
                        continue
                    except Exception as e:
                        print(f"MJPEG fallback failed: {e}")
                        break
            else:
                break

        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        display_text = DISPLAY_ON_UNSTABLE
        confidence_text = ""
        
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            
            # Draw landmarks
            frame = draw_landmarks_on_frame(frame, hand, mp_hands)

            # Detect handedness (MediaPipe provides Left/Right) and extract features
            handedness_label = None
            try:
                if result.multi_handedness:
                    handedness_label = result.multi_handedness[0].classification[0].label
            except Exception:
                handedness_label = None

            # Extract normalized features
            features = extract_hand_landmarks(hand)

            # If model was trained on right-hand only, mirror left-hand features so they match
            # (flip the x-coordinates in the flattened [x,y,z,...] feature vector)
            if handedness_label and handedness_label.lower().startswith("l"):
                for i in range(0, len(features), 3):
                    features[i] = -features[i]
            
            # Make prediction
            probs = model.predict_proba([features])[0]
            pred = model.classes_[np.argmax(probs)]
            confidence = np.max(probs)

            # Add to smoothing buffer and compute majority
            prediction_buffer.append(pred)
            most_common, count = Counter(prediction_buffer).most_common(1)[0]

            # If the buffer is full (i.e. we have STABLE_FRAMES samples) and the majority
            # agrees, and the classifier is confident, accept the gesture.
            if len(prediction_buffer) == STABLE_FRAMES and count >= STABLE_FRAMES and confidence > CONFIDENCE_THRESHOLD:
                display_text = most_common

            # Update helpers for debugging / logging
            last_prediction = pred
            stable_count = count
            confidence_text = f"Confidence: {confidence:.2f}"

            # Print compact debug info
            print(f"pred={pred:10} | top={most_common:10}({count}/{STABLE_FRAMES}) | conf={confidence:.2f} | buffer={list(prediction_buffer)}")
        
        else:
            # No hand detected: clear smoothing buffer so previous gestures don't stick
            prediction_buffer.clear()
            stable_count = 0
            last_prediction = ""

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
