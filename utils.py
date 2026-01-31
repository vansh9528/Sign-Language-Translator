"""
Utility functions for Sign Language Translator
"""

import cv2
import mediapipe as mp
import numpy as np
from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
    STATIC_IMAGE_MODE,
    CAMERA_INDEX,
    CAMERA_BACKEND
)


def initialize_hand_detector():
    """Initialize MediaPipe hand detector with configured settings."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        static_image_mode=STATIC_IMAGE_MODE
    )
    return hands, mp_hands


def initialize_camera(backend="DSHOW"):
    """
    Initialize camera with specified backend.
    
    Args:
        backend (str): Camera backend ("DSHOW", "V4L2", "AVFOUNDATION")
    
    Returns:
        cv2.VideoCapture: Camera object
    """
    backend_map = {
        "DSHOW": cv2.CAP_DSHOW,
        "V4L2": cv2.CAP_V4L2,
        "AVFOUNDATION": cv2.CAP_AVFOUNDATION,
        "MSMF": cv2.CAP_MSMF,
    }
    
    backend_id = backend_map.get(backend, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(CAMERA_INDEX, backend_id)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera with {backend} backend")
    
    return cap


def extract_hand_landmarks(hand_landmarks):
    """
    Extract normalized hand landmarks (relative to wrist, scale-invariant).
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
    
    Returns:
        list: Flattened list of 21 landmarks × 3 coordinates (x, y, z) = 63 features
    """
    # Step 1: Use wrist (landmark 0) as origin
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    base_z = hand_landmarks.landmark[0].z

    # Step 2: Compute relative coordinates and find max distance
    rel_points = []
    for lm in hand_landmarks.landmark:
        rel_x = lm.x - base_x
        rel_y = lm.y - base_y
        rel_z = lm.z - base_z
        rel_points.append((rel_x, rel_y, rel_z))

    # Step 3: Scale by max distance (makes it scale-invariant)
    max_dist = max(
        (abs(x) + abs(y) + abs(z)) for (x, y, z) in rel_points
    )
    
    # Avoid division by zero
    if max_dist < 0.001:
        max_dist = 1.0

    # Step 4: Normalize and flatten
    features = []
    for (x, y, z) in rel_points:
        features.extend([
            x / max_dist,
            y / max_dist,
            z / max_dist
        ])
    
    return features


def draw_landmarks_on_frame(frame, hand_landmarks, mp_hands):
    """
    Draw hand skeleton on frame.
    
    Args:
        frame: OpenCV frame
        hand_landmarks: MediaPipe hand landmarks
        mp_hands: MediaPipe hands module
    
    Returns:
        frame: Modified frame with drawn landmarks
    """
    mp_draw = mp.solutions.drawing_utils
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame


def add_text_to_frame(frame, text, position=(30, 40), font_scale=1, 
                      color=(0, 255, 0), thickness=2):
    """
    Add text to frame.
    
    Args:
        frame: OpenCV frame
        text: Text to add
        position: (x, y) position
        font_scale: Font size
        color: BGR color tuple
        thickness: Text thickness
    
    Returns:
        frame: Modified frame with text
    """
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    return frame
