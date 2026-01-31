"""
Data Collection Script for Sign Language Translator

This script collects hand landmark data for training the gesture recognition model.
Change the gesture variable to collect different gestures.

Usage:
    python collect_data.py
"""

import cv2
import csv
import os
from config import (
    SAMPLES_PER_GESTURE,
    DATASET_FOLDER,
    CAMERA_BACKEND
)
from utils import (
    initialize_hand_detector,
    initialize_camera,
    extract_hand_landmarks,
    draw_landmarks_on_frame,
    add_text_to_frame
)


def collect_gesture_data(gesture_name, num_samples):
    """
    Collect hand landmark data for a specific gesture.
    
    Args:
        gesture_name (str): Name of the gesture (e.g., "BAD", "GOOD")
        num_samples (int): Number of samples to collect
    """
    # Initialize detector and camera
    hands, mp_hands = initialize_hand_detector()
    cap = initialize_camera(CAMERA_BACKEND)
    
    # Create dataset folder if it doesn't exist
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    file_path = f"{DATASET_FOLDER}/{gesture_name}.csv"
    
    print(f"\n{'='*50}")
    print(f"Collecting gesture: {gesture_name}")
    print(f"Target samples: {num_samples}")
    print(f"Output file: {file_path}")
    print(f"{'='*50}")
    print("Press ESC to stop early\n")
    
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        count = 0
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                
                # Extract normalized landmarks
                features = extract_hand_landmarks(hand)
                
                # Write to CSV
                writer.writerow(features)
                count += 1
                
                # Draw on frame
                frame = draw_landmarks_on_frame(frame, hand, mp_hands)
                frame = add_text_to_frame(
                    frame,
                    f"{gesture_name}: {count}/{num_samples}",
                    position=(20, 40),
                    font_scale=1
                )
                
                print(f"✓ {gesture_name} sample {count}/{num_samples}")
            
            # Draw instruction text
            frame = add_text_to_frame(
                frame,
                "Show your hand clearly",
                position=(20, 80),
                font_scale=0.7,
                color=(255, 255, 0)
            )
            
            cv2.imshow("Collecting Gesture Data", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Saved {count} samples to {file_path}")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("SIGN LANGUAGE TRANSLATOR - DATA COLLECTION")
    print("="*50)
    
    # Ask user for gesture name
    gesture = input("\nEnter gesture name (e.g., BAD, GOOD, YES, NO, PEACE, THANK YOU): ").strip().upper()
    
    if not gesture:
        print("❌ Gesture name cannot be empty!")
        exit(1)
    
    # Collect data
    collect_gesture_data(gesture, SAMPLES_PER_GESTURE)
    
    print("\n✓ Data collection complete!")
