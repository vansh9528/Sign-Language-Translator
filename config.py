"""
Configuration settings for Sign Language Translator
"""

# Camera settings
CAMERA_INDEX = 0
CAMERA_BACKEND = "DSHOW"  # Windows: DSHOW, Linux: V4L2, Mac: AVFOUNDATION

# ESP32-CAM (MJPEG) stream settings
# Set USE_ESP32 = True to use ESP32 stream instead of the local webcam.
# You can also set CAMERA_INDEX to the stream URL directly (e.g. "http://192.168.1.100:81/stream").
USE_ESP32 = True
ESP32_STREAM_URL = "http://192.168.204.95:81/stream"
ESP32_RETRY_COUNT = 5
ESP32_RETRY_DELAY = 1.0  # seconds between retry attempts

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
STABLE_FRAMES = 3  # Frames needed for stable prediction (lowered for faster feedback during testing)
CONFIDENCE_THRESHOLD = 0.35  # Minimum confidence to display gesture (lowered for testing)
DISPLAY_ON_UNSTABLE = "..."  # What to show when not confident

# Display settings
WINDOW_NAME = "Sign Language Translator"
FONT = "HERSHEY_SIMPLEX"
FONT_SCALE = 1
FONT_COLOR = (0, 255, 0)  # BGR format (Green)
FONT_THICKNESS = 2
FLIP_FRAME = False  # For ESP32 stream prefer no horizontal flip; set True for webcam mirroring

# Gesture names (must match your CSV filenames without .csv)
GESTURES = [
    "BAD",
    "GOOD", 
    "YES",
    "NO",
    "PEACE",
    "THANK YOU"
]
