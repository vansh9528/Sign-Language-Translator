"""
Batch Collection Helper for PEACE_GOOD Gesture

Quick script to collect PEACE_GOOD samples without manual prompts.
"""

import subprocess
import sys
from pathlib import Path


def check_csv_exists(gesture_name):
    """Check if gesture CSV already exists."""
    csv_file = Path(f"landmark_dataset/{gesture_name}_VIDEO.csv")
    if csv_file.exists():
        size = csv_file.stat().st_size / 1024  # KB
        return True, size
    return False, 0


def run_collection():
    """Run PEACE_GOOD collection."""
    
    print("\n" + "="*60)
    print("PEACE_GOOD VIDEO GESTURE COLLECTION")
    print("="*60)
    print("\nThis will collect PEACE_GOOD gesture in video format:")
    print("  1. PEACE sign (hand on side)")
    print("  2. GOOD sign (thumbs up)")
    print("\n" + "-"*60)
    
    gesture_name = "PEACE_GOOD"
    exists, size = check_csv_exists(gesture_name)
    
    if exists:
        print(f"\n⚠️  WARNING: {gesture_name}_VIDEO.csv already exists ({size:.1f} KB)")
        print("Collection will ADD samples to existing file.\n")
        response = input("Continue? (yes/no): ").strip().lower()
        if response != "yes":
            print("❌ Cancelled")
            return
    
    # Get number of samples
    while True:
        try:
            num_samples = int(input("\nHow many samples to collect? (default 40): ") or "40")
            if num_samples <= 0:
                print("❌ Must be > 0")
                continue
            break
        except ValueError:
            print("❌ Invalid number")
    
    print(f"\n✓ Collection setup:")
    print(f"  Gesture: {gesture_name}")
    print(f"  Samples: {num_samples}")
    print(f"  Method: Video Sequences (5-frame temporal)")
    print(f"  Output: landmark_dataset/{gesture_name}_VIDEO.csv")
    
    print("\n" + "-"*60)
    print("PERFORMANCE TIPS:")
    print("  • Position hand 30-50cm from camera")
    print("  • Hold PEACE on side (frames 1-2)")
    print("  • Transition smoothly to GOOD/thumbs-up (frames 3-5)")
    print("  • Press SPACE to record, release when done")
    print("  • Perform naturally - ~2 seconds per sample")
    print("-"*60)
    
    response = input("\nReady? Press Enter to start...")
    
    # Run collection
    print("\nLaunching collection interface...")
    print("(This window will open the video capture)\n")
    
    # Create inline Python to run collection
    collection_code = f"""
import sys
sys.path.insert(0, '.')

from collect_video_data import collect_video_gesture_data

collect_video_gesture_data(
    gesture_name="{gesture_name}",
    num_samples={num_samples},
    sequence_length=5
)
"""
    
    try:
        exec(collection_code)
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        return
    
    # Verify collection
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    
    exists, size = check_csv_exists(gesture_name)
    if exists:
        print(f"✓ File saved: {gesture_name}_VIDEO.csv ({size:.1f} KB)")
        print(f"\nNext steps:")
        print(f"  1. Train model: python train_video_model.py")
        print(f"  2. Test live: python live_realtime_video.py")
        print(f"  3. Perform PEACE_GOOD and watch it predict!")
    else:
        print(f"❌ File not found - collection may have failed")


if __name__ == "__main__":
    run_collection()
