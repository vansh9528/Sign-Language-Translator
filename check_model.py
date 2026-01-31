"""
Check what gestures are trained in each model.
"""

import pickle
import os

def check_model(model_path):
    """Check classes in trained model."""
    if not os.path.exists(model_path):
        print(f"❌ {model_path} not found\n")
        return
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Get the SVM model and its classes
        if hasattr(model, 'classes_'):
            classes = model.classes_
        elif hasattr(model, 'named_steps'):
            # Pipeline with SVC
            classes = model.named_steps['classifier'].classes_
        else:
            classes = "Unknown"
        
        print(f"✓ {model_path}")
        print(f"  Classes: {sorted(classes)}")
        print(f"  Count: {len(classes)}\n")
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}\n")


print("="*60)
print("TRAINED GESTURES CHECK")
print("="*60 + "\n")

check_model("gesture_model.pkl")
check_model("gesture_model_video.pkl")

print("-"*60)
print("WHICH TO USE:")
print("-"*60)
print("• live_realtime.py → uses gesture_model.pkl")
print("• live_realtime_video.py → uses gesture_model_video.pkl")
print("\nIf HELLO not in live_realtime.py output:")
print("  Run: python live_realtime_video.py")
print("="*60)
