"""
INDEX OF ALL PROJECT FILES
Sign Language Translator - Standardized Version

Quick navigation guide for the entire project.
"""

# Use UTF-8 for console output
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║         SIGN LANGUAGE TRANSLATOR - FILE INDEX & GUIDE                     ║
╚════════════════════════════════════════════════════════════════════════════╝

📋 START HERE
═════════════════════════════════════════════════════════════════════════════

1. 🚀 FIRST TIME SETUP
   → Read: QUICKSTART.py (print it: python QUICKSTART.py)
   → Follow the 4 steps to get started

2. 📚 COMPLETE GUIDE
   → Read: README.md (all project details)
   → Read: STANDARDIZATION_COMPLETE.md (what was improved)

3. 🛠️  EXTEND THE PROJECT
   → Read: DEVELOPMENT_GUIDE.md (how to add features)


🔧 CORE APPLICATION FILES
═════════════════════════════════════════════════════════════════════════════

collect_data.py
  • Collects hand gesture training data
  • Run: python collect_data.py
  • Interactive - asks for gesture name
  • Collects 250 samples per gesture (configurable)

train_landmark_model.py
  • Trains SVM model on collected gesture data
  • Run: python train_landmark_model.py
  • Loads all CSVs from landmark_dataset/
  • Displays accuracy metrics
  • Saves model as gesture_model.pkl

live_realtime.py
  • Real-time gesture recognition using trained model
  • Run: python live_realtime.py
  • Live video with gesture predictions
  • Shows confidence scores
  • Press ESC to exit


⚙️  CONFIGURATION & UTILITIES
═════════════════════════════════════════════════════════════════════════════

config.py
  • Centralized configuration for entire project
  • All settings in one place
  • No code changes needed - just edit config.py
  • Sections:
    - Camera settings (index, backend)
    - MediaPipe settings (confidence, tracking)
    - Data collection (samples, folder)
    - Model hyperparameters (kernel, C, gamma)
    - Real-time prediction (thresholds, smoothing)
    - Display settings (colors, fonts)

utils.py
  • Reusable utility functions
  • Used by all three main scripts
  • Functions:
    - initialize_hand_detector()
    - initialize_camera(backend)
    - extract_hand_landmarks(hand)
    - draw_landmarks_on_frame(frame, hand, mp_hands)
    - add_text_to_frame(frame, text, ...)
  • All include error handling and documentation


📚 DOCUMENTATION FILES
═════════════════════════════════════════════════════════════════════════════

README.md ⭐ START HERE
  • Complete project documentation
  • Features overview
  • Installation instructions
  • Detailed usage guide
  • Configuration reference
  • Troubleshooting section
  • Technical details
  • ~500 lines of comprehensive guide

QUICKSTART.py
  • Print with: python QUICKSTART.py
  • 4-step quick start guide
  • Basic troubleshooting
  • File structure overview
  • Run times: ~2 minutes to read

PROJECT_STANDARDIZATION.md
  • Summary of all improvements made
  • Before/after comparison
  • File structure breakdown
  • Best practices implemented
  • Read time: 10 minutes

DEVELOPMENT_GUIDE.md
  • How to extend and modify the project
  • Example: adding a new feature
  • Modifying hyperparameters
  • Adding new gestures
  • ESP32 integration tips
  • Performance optimization
  • Common modifications reference

STANDARDIZATION_COMPLETE.md
  • Final summary of the standardization
  • Quick start (3 steps)
  • Configuration overview
  • Quality checklist
  • Maturity assessment


📋 CONFIGURATION & DEPENDENCY FILES
═════════════════════════════════════════════════════════════════════════════

requirements.txt
  • All Python package dependencies
  • Pinned to specific versions
  • Install with: pip install -r requirements.txt
  • Packages:
    - opencv-python (computer vision)
    - mediapipe (hand detection)
    - numpy (numerical computing)
    - pandas (data handling)
    - scikit-learn (machine learning)

config.py
  • See above in "CONFIGURATION & UTILITIES"

.gitignore
  • Git ignore rules
  • Excludes cache, models, datasets
  • Keeps repository clean
  • Standard Python project patterns


📊 DATA & MODELS
═════════════════════════════════════════════════════════════════════════════

landmark_dataset/ (folder)
  • Created automatically by collect_data.py
  • Contains CSV files for each gesture
  • Files:
    - BAD.csv (default gesture)
    - GOOD.csv (default gesture)
    - YES.csv (default gesture)
    - NO.csv (default gesture)
    - PEACE.csv (default gesture)
    - THANK YOU.csv (default gesture)
    - Or your custom gestures
  • Each row = 1 hand sample (63 features)
  • Format: comma-separated landmarks

gesture_model.pkl
  • Created by train_landmark_model.py
  • Trained SVM model (machine learning)
  • Loaded and used by live_realtime.py
  • ~100-500 KB file size
  • Binary format (pickle)


🧪 TESTING & DEBUG UTILITIES
═════════════════════════════════════════════════════════════════════════════

test_camera.py
  • Tests if camera is accessible
  • Run: python test_camera.py
  • Output: Shows which camera index works

test_backends.py
  • Tests different camera backends
  • Run: python test_backends.py
  • Finds which backend works on your system
  • (Used to fix Windows camera issues)

debug_test.py
  • Tests hand detection with low thresholds
  • Run: python debug_test.py
  • Shows if MediaPipe can detect hands
  • Counts total detections

debug_detailed.py
  • Detailed debug output for development
  • For troubleshooting specific issues


📁 PROJECT STRUCTURE EXPLANATION
═════════════════════════════════════════════════════════════════════════════

Root directory (Sign-Language-Translator/)
│
├── Application Scripts (what to run)
│   ├── collect_data.py           → Run first to collect data
│   ├── train_landmark_model.py    → Run second to train
│   └── live_realtime.py          → Run third to test
│
├── Core Modules (used by scripts)
│   ├── config.py                 → Settings
│   └── utils.py                  → Utilities
│
├── Documentation (read first!)
│   ├── README.md                 ⭐ START
│   ├── QUICKSTART.py             → Quick guide
│   ├── STANDARDIZATION_COMPLETE.md
│   ├── DEVELOPMENT_GUIDE.md
│   └── PROJECT_STANDARDIZATION.md
│
├── Dependencies
│   ├── requirements.txt           → pip install -r requirements.txt
│   └── .gitignore               → Git rules
│
├── Data (created by scripts)
│   ├── landmark_dataset/         → Training CSVs
│   ├── gesture_model.pkl         → Trained model
│   └── __pycache__/             → Python cache
│
└── Testing Utilities
    ├── test_camera.py
    ├── test_backends.py
    ├── debug_test.py
    └── debug_detailed.py


🎯 TYPICAL WORKFLOW
═════════════════════════════════════════════════════════════════════════════

1. SETUP (First time only)
   pip install -r requirements.txt

2. COLLECT DATA
   python collect_data.py
   → BAD (collect 250 samples)
   → GOOD (collect 250 samples)
   → YES (collect 250 samples)
   → ... (collect more gestures)

3. TRAIN MODEL
   python train_landmark_model.py
   → Trains on all collected data
   → Shows accuracy (~85-95%)
   → Saves gesture_model.pkl

4. TEST REAL-TIME
   python live_realtime.py
   → Live video
   → Show gestures
   → See predictions
   → Press ESC to exit

5. CUSTOMIZE (Optional)
   → Edit config.py for different settings
   → Collect more data for better accuracy
   → Add new gestures
   → Retrain model


❓ WHICH FILE SHOULD I READ FIRST?
═════════════════════════════════════════════════════════════════════════════

I just want to... → Read this...
─────────────────────────────────────────────────────────────────────────────
...get started     → QUICKSTART.py (or run: python QUICKSTART.py)
...understand      → README.md (comprehensive guide)
...understand the  → PROJECT_STANDARDIZATION.md or
  improvements       STANDARDIZATION_COMPLETE.md
...extend the code → DEVELOPMENT_GUIDE.md
...fix an error    → README.md (Troubleshooting section)
...see the         → config.py (read the comments!)
  configuration


🔍 CONFIGURATION QUICK REFERENCE
═════════════════════════════════════════════════════════════════════════════

To change... → Edit this in config.py
─────────────────────────────────────────────────────────────────────────────
Camera                  → CAMERA_INDEX, CAMERA_BACKEND
Hand detection          → MIN_DETECTION_CONFIDENCE
Samples per gesture     → SAMPLES_PER_GESTURE
Model type              → SVM_KERNEL, SVM_C
Gesture prediction      → STABLE_FRAMES, CONFIDENCE_THRESHOLD
Display/UI              → FONT_SCALE, FONT_COLOR, WINDOW_NAME


⚡ QUICK COMMANDS
═════════════════════════════════════════════════════════════════════════════

# Setup
pip install -r requirements.txt

# Collect data for gesture "BAD" (250 samples)
python collect_data.py

# Train model on all gesture data
python train_landmark_model.py

# Run real-time recognition
python live_realtime.py

# Show quick start guide
python QUICKSTART.py

# Test camera access
python test_camera.py

# Find working camera backend
python test_backends.py

# Debug hand detection
python debug_test.py


🆘 COMMON PROBLEMS
═════════════════════════════════════════════════════════════════════════════

Problem → Solution
─────────────────────────────────────────────────────────────────────────────
Camera not working    → See README.md Troubleshooting
No hand detected      → See README.md Troubleshooting
Low accuracy          → See README.md Troubleshooting
How to add gesture?   → See DEVELOPMENT_GUIDE.md
How to customize?     → Edit config.py


📞 NEED HELP?
═════════════════════════════════════════════════════════════════════════════

1. Check README.md (most comprehensive guide)
2. Check QUICKSTART.py (quick walkthrough)
3. Check DEVELOPMENT_GUIDE.md (for customization)
4. Check code comments (many details in code)
5. Read docstrings (python help(function_name))


✨ YOU'RE ALL SET!
═════════════════════════════════════════════════════════════════════════════

Your project is:
✅ Well-organized with clear structure
✅ Fully documented with multiple guides
✅ Easy to use with simple commands
✅ Easy to customize via config.py
✅ Ready for production use

👉 Start with README.md or QUICKSTART.py!

═════════════════════════════════════════════════════════════════════════════
""")
