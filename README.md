# Sign Language Translator

A real-time sign language gesture recognition system using MediaPipe hand detection and machine learning (SVM).

## Features

- 🎥 Real-time hand gesture detection using MediaPipe
- 🤖 SVM-based gesture classification with scale-invariant normalization
- 📊 Multi-gesture support (customizable)
- 📁 Organized project structure with configuration management
- 🎯 High accuracy with temporal smoothing for stability

## Project Structure

```
Sign-Language-Translator/
├── config.py                      # Centralized configuration
├── utils.py                       # Utility functions (reusable)
├── collect_data.py               # Data collection script
├── train_landmark_model.py        # Model training script
├── live_realtime.py              # Real-time prediction script
├── landmark_dataset/             # Gesture training data (auto-created)
│   ├── BAD.csv
│   ├── GOOD.csv
│   ├── YES.csv
│   └── ...
├── gesture_model.pkl             # Trained SVM model (auto-created)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### 1. Clone or download the project

```bash
cd Sign-Language-Translator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install opencv-python==4.8.1.78 mediapipe==0.10.14 numpy==1.26.4 pandas==2.1.3 scikit-learn==1.3.2
```

## Usage

### Step 1: Collect Training Data

Collect hand landmark data for each gesture you want to recognize:

```bash
python collect_data.py
```

**Instructions:**
1. When prompted, enter the gesture name (e.g., `BAD`, `GOOD`, `YES`, `NO`, `PEACE`, `THANK YOU`)
2. Show your hand clearly to the camera
3. Vary your distance and position (near, medium, far) for better generalization
4. The script will collect 250 samples by default (configurable in `config.py`)
5. Press `ESC` to stop early

**Repeat for each gesture you want to recognize.**

### Step 2: Train the Model

After collecting data for all gestures:

```bash
python train_landmark_model.py
```

This will:
- Load all gesture data from `landmark_dataset/`
- Split into 80% training / 20% testing
- Train an SVM model with StandardScaler preprocessing
- Display accuracy and classification metrics
- Save the model as `gesture_model.pkl`

### Step 3: Run Real-Time Prediction

```bash
python live_realtime.py
```

**Instructions:**
1. Show your hand to the camera
2. Perform gestures from your trained set
3. The recognized gesture will appear on screen
4. Press `ESC` to exit

## Configuration

Edit `config.py` to customize:

### Camera Settings
- `CAMERA_INDEX`: Which camera to use (0 = default)
- `CAMERA_BACKEND`: "DSHOW" (Windows), "V4L2" (Linux), "AVFOUNDATION" (Mac)

### MediaPipe Settings
- `MIN_DETECTION_CONFIDENCE`: Lower = more detections (0.0-1.0)
- `MIN_TRACKING_CONFIDENCE`: Lower = smoother tracking (0.0-1.0)

### Data Collection
- `SAMPLES_PER_GESTURE`: Number of samples per gesture (default: 250)
- `DATASET_FOLDER`: Where to save training data

### Model Training
- `SVM_KERNEL`: "rbf", "linear", "poly"
- `SVM_C`: Regularization parameter (higher = stricter)
- `TEST_SIZE`: Train/test split ratio (0.2 = 20% test)

### Real-Time Prediction
- `STABLE_FRAMES`: Frames needed for stable prediction (default: 8)
- `CONFIDENCE_THRESHOLD`: Minimum confidence to display (0.0-1.0)

## How It Works

### 1. Feature Extraction (Landmark Normalization)

All hand gestures are extracted as **63-dimensional vectors** (21 landmarks × 3 coordinates):

1. **Position Relative to Wrist**: All landmarks are computed relative to the wrist (landmark 0) to make position-invariant
2. **Scale Normalization**: Features are divided by the max L1 distance to make scale-invariant

This ensures the same gesture is recognized regardless of:
- Where your hand is in the frame
- How far or close your hand is from the camera
- Hand size

### 2. Model Training

- **Preprocessing**: StandardScaler normalizes features for SVM
- **Algorithm**: Support Vector Machine (SVM) with RBF kernel
- **Class Weight**: Balanced weighting for fair multi-class classification

### 3. Real-Time Prediction

- Hand detected → Landmarks extracted → Features normalized
- SVM predicts gesture + confidence score
- **Temporal Smoothing**: Requires 8 consecutive frames of same prediction before displaying
- Reduces false positives and flicker

## Supported Gestures

Default setup includes:
- `BAD`
- `GOOD`
- `YES`
- `NO`
- `PEACE`
- `THANK YOU`

To add custom gestures, just collect data with a new name in `collect_data.py`.

## Troubleshooting

### Camera not working

- Check `CAMERA_BACKEND` in `config.py`
- Try different backends: "DSHOW" (Windows), "V4L2" (Linux)
- Run `python test_camera.py` to test camera access

### No hand detected

- Ensure good lighting
- Show your full hand clearly
- Lower `MIN_DETECTION_CONFIDENCE` in `config.py` (try 0.1)
- Move closer to camera

### Model accuracy is low

- Collect more diverse data (vary distance, angle, lighting)
- Ensure gestures are distinct from each other
- Increase `SAMPLES_PER_GESTURE` in `config.py`
- Retrain the model with new data

### Model not found error when running live_realtime.py

- First train the model: `python train_landmark_model.py`
- Ensure `gesture_model.pkl` exists in the project folder

## Performance Tips

1. **Better accuracy**: Collect 500+ samples per gesture
2. **Faster inference**: Reduce `MIN_DETECTION_CONFIDENCE` (trades accuracy for speed)
3. **Smoother predictions**: Increase `STABLE_FRAMES` (trades responsiveness for stability)

## Requirements

- Python 3.8+
- OpenCV 4.8+
- MediaPipe 0.10+
- scikit-learn 1.3+
- NumPy, Pandas

## Technical Details

### Feature Vector Size
- 21 hand landmarks × 3 coordinates (x, y, z) = 63 features

### Model Size
- SVM model: ~100KB - 1MB (very lightweight)

### Inference Speed
- ~30-50 FPS on modern CPU (i7+)
- Faster on GPU-accelerated systems

## License

This project uses:
- MediaPipe: [Apache 2.0](https://github.com/google/mediapipe)
- OpenCV: [Apache 2.0](https://opencv.org/license/)
- scikit-learn: [BSD 3-Clause](https://github.com/scikit-learn/scikit-learn)

## Contributing

Feel free to improve this project! Some ideas:
- Add data augmentation
- Implement deep learning models (CNN, RNN)
- Add ESP32 integration for edge deployment
- Improve UI/UX

## Author

Created for sign language recognition using computer vision.

## Support

For issues or questions, check the code comments or review the configuration options in `config.py`.

---

**Happy recognizing! 🎉**
