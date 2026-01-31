"""
Configuration settings for Sign Language Translator
"""

# Camera settings
CAMERA_INDEX = 0
CAMERA_BACKEND = "DSHOW"  # Windows: DSHOW, Linux: V4L2, Mac: AVFOUNDATION

# MediaPipe Hand Detection settings
MIN_DETECTION_CONFIDENCE = 0.1
MIN_TRACKING_CONFIDENCE = 0.1
MAX_NUM_HANDS = 1
STATIC_IMAGE_MODE = False

# Data collection settings
SAMPLES_PER_GESTURE = 250
DATASET_FOLDER = "landmark_dataset"

# Model settings
MODEL_PATH = "gesture_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# SVM Model hyperparameters
SVM_KERNEL = "rbf"
SVM_C = 10
SVM_GAMMA = "scale"
SVM_PROBABILITY = True

# Real-time prediction settings
STABLE_FRAMES = 8  # Frames needed for stable prediction
CONFIDENCE_THRESHOLD = 0.45  # Minimum confidence to display gesture
DISPLAY_ON_UNSTABLE = "..."  # What to show when not confident

# Display settings
WINDOW_NAME = "Sign Language Translator"
FONT = "HERSHEY_SIMPLEX"
FONT_SCALE = 1
FONT_COLOR = (0, 255, 0)  # BGR format (Green)
FONT_THICKNESS = 2

# Gesture names (must match your CSV filenames without .csv)
GESTURES = [
    "BAD",
    "GOOD", 
    "YES",
    "NO",
    "PEACE",
    "THANK YOU"
]
