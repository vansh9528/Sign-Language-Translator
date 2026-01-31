"""
Video-Based Gesture Training

This script collects hand gesture training data from video sequences,
capturing motion and temporal dynamics of gestures.

Usage:
    python collect_video_data.py
"""

import cv2
import mediapipe as mp
import csv
import os
from collections import deque
from config import (
    SAMPLES_PER_GESTURE,
    DATASET_FOLDER,
    CAMERA_BACKEND,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)
from utils import (
    initialize_hand_detector,
    initialize_camera,
    extract_hand_landmarks,
    draw_landmarks_on_frame,
    add_text_to_frame
)


def extract_temporal_features(landmarks_sequence, num_frames=5):
    """
    Extract temporal features from a sequence of hand landmarks.
    
    Captures motion by computing differences between consecutive frames.
    
    Args:
        landmarks_sequence: List of landmark vectors
        num_frames: Number of frames to use for temporal features
    
    Returns:
        list: Flattened temporal feature vector
    """
    if len(landmarks_sequence) < num_frames:
        # Pad with zeros if not enough frames
        landmarks_sequence = landmarks_sequence + [
            [0] * 63 for _ in range(num_frames - len(landmarks_sequence))
        ]
    
    # Get last N frames
    recent_frames = landmarks_sequence[-num_frames:]
    
    # Compute temporal features
    temporal_features = []
    
    # 1. Stack current landmarks
    for frame in recent_frames:
        temporal_features.extend(frame)
    
    # 2. Compute velocity (frame-to-frame differences)
    for i in range(1, len(recent_frames)):
        velocity = [
            recent_frames[i][j] - recent_frames[i-1][j]
            for j in range(len(recent_frames[i]))
        ]
        temporal_features.extend(velocity)
    
    # 3. Compute acceleration (change in velocity)
    if len(recent_frames) > 2:
        for i in range(2, len(recent_frames)):
            accel = [
                (recent_frames[i][j] - recent_frames[i-1][j]) -
                (recent_frames[i-1][j] - recent_frames[i-2][j])
                for j in range(len(recent_frames[i]))
            ]
            temporal_features.extend(accel)
    
    return temporal_features


def collect_video_gesture_data(gesture_name, num_samples, sequence_length=5):
    """
    Collect video-based gesture data.
    
    Args:
        gesture_name (str): Name of the gesture
        num_samples (int): Number of gesture sequences to collect
        sequence_length (int): Number of frames per gesture sequence
    """
    # Initialize
    hands, mp_hands = initialize_hand_detector()
    cap = initialize_camera(CAMERA_BACKEND)
    
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    file_path = f"{DATASET_FOLDER}/{gesture_name}_VIDEO.csv"
    
    print(f"\n{'='*60}")
    print(f"COLLECTING VIDEO GESTURE DATA")
    print(f"{'='*60}")
    print(f"Gesture: {gesture_name}")
    print(f"Target samples: {num_samples}")
    print(f"Sequence length: {sequence_length} frames")
    print(f"Output file: {file_path}")
    print(f"{'='*60}")
    print("""
Instructions:
1. When you see "RECORDING", start performing the gesture
2. Hold the gesture for ~1-2 seconds (while recording)
3. Press SPACE to save the gesture sequence
4. Press ESC to quit early
    """)
    
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        count = 0
        frame_buffer = deque(maxlen=sequence_length)
        is_recording = False
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            # Display status
            status_text = "RECORDING" if is_recording else "READY"
            status_color = (0, 255, 0) if is_recording else (255, 255, 0)
            
            frame = add_text_to_frame(
                frame,
                status_text,
                position=(50, 50),
                font_scale=1.5,
                color=status_color,
                thickness=3
            )
            
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                frame = draw_landmarks_on_frame(frame, hand, mp_hands)
                
                # Extract landmarks
                features = extract_hand_landmarks(hand)
                
                # Add to buffer if recording
                if is_recording:
                    frame_buffer.append(features)
                    
                    # Show buffer progress
                    progress = f"Recording: {len(frame_buffer)}/{sequence_length}"
                    frame = add_text_to_frame(
                        frame,
                        progress,
                        position=(50, 100),
                        font_scale=1,
                        color=(0, 255, 0)
                    )
            else:
                if is_recording:
                    frame = add_text_to_frame(
                        frame,
                        "Hand lost! Recording stopped",
                        position=(50, 100),
                        font_scale=0.8,
                        color=(0, 0, 255)
                    )
            
            # Show collected count
            frame = add_text_to_frame(
                frame,
                f"Collected: {count}/{num_samples}",
                position=(50, 150),
                font_scale=1,
                color=(255, 255, 0)
            )
            
            # Show instructions
            frame = add_text_to_frame(
                frame,
                "SPACE: Start/Save | ESC: Quit",
                position=(50, frame.shape[0] - 30),
                font_scale=0.7,
                color=(200, 200, 200)
            )
            
            cv2.imshow("Video Gesture Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE
                if not is_recording:
                    # Start recording
                    is_recording = True
                    frame_buffer.clear()
                    print(f"Recording started...")
                elif len(frame_buffer) == sequence_length:
                    # Save recorded sequence
                    temporal_features = extract_temporal_features(list(frame_buffer), sequence_length)
                    writer.writerow(temporal_features)
                    count += 1
                    is_recording = False
                    frame_buffer.clear()
                    print(f"✓ {gesture_name} sample {count}/{num_samples} saved")
                else:
                    print(f"❌ Need {sequence_length} frames, got {len(frame_buffer)}")
            
            elif key == 27:  # ESC
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Saved {count} video sequences to {file_path}")
    return count


def collect_frame_by_frame(gesture_name, num_samples):
    """
    Simpler approach: Collect individual frames.
    Each frame is treated as a separate sample (like images).
    
    Args:
        gesture_name (str): Name of the gesture
        num_samples (int): Number of frames to collect
    """
    hands, mp_hands = initialize_hand_detector()
    cap = initialize_camera(CAMERA_BACKEND)
    
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    file_path = f"{DATASET_FOLDER}/{gesture_name}.csv"
    
    print(f"\n{'='*60}")
    print(f"COLLECTING FRAME-BY-FRAME GESTURE DATA")
    print(f"{'='*60}")
    print(f"Gesture: {gesture_name}")
    print(f"Target samples: {num_samples}")
    print(f"Output file: {file_path}")
    print(f"{'='*60}")
    print("""
Instructions:
1. Show your hand clearly
2. Perform the gesture
3. Each frame is automatically saved when hand is detected
4. Press ESC to quit
    """)
    
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        count = 0
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                frame = draw_landmarks_on_frame(frame, hand, mp_hands)
                
                features = extract_hand_landmarks(hand)
                writer.writerow(features)
                count += 1
                
                frame = add_text_to_frame(
                    frame,
                    f"{gesture_name}: {count}/{num_samples}",
                    position=(20, 40),
                    font_scale=1
                )
                
                print(f"✓ {gesture_name} sample {count}/{num_samples}")
            else:
                frame = add_text_to_frame(
                    frame,
                    "Show your hand",
                    position=(20, 40),
                    font_scale=1,
                    color=(0, 0, 255)
                )
            
            cv2.imshow("Frame Collection", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Saved {count} frames to {file_path}")
    return count


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VIDEO GESTURE DATA COLLECTION")
    print("="*60)
    
    # Ask user for collection method
    print("\nChoose collection method:")
    print("1. Video sequences (captures motion) - RECOMMENDED")
    print("2. Frame-by-frame (simpler)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        gesture = input("\nEnter gesture name: ").strip().upper()
        if gesture:
            collect_video_gesture_data(gesture, SAMPLES_PER_GESTURE, sequence_length=5)
            print("\n✓ Video gesture collection complete!")
    
    elif choice == "2":
        gesture = input("\nEnter gesture name: ").strip().upper()
        if gesture:
            collect_frame_by_frame(gesture, SAMPLES_PER_GESTURE)
            print("\n✓ Frame collection complete!")
    
    else:
        print("Exited")
