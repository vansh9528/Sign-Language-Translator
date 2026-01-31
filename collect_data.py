import cv2
import mediapipe as mp
import csv
import os

# ================== USER SETTINGS ==================
GESTURE = "PEACE"        # 👈 yahan gesture ka naam likho
SAMPLES = 250         # 👈 har gesture ke samples
# ===================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Dataset folder
os.makedirs("landmark_dataset", exist_ok=True)
file_path = f"landmark_dataset/{GESTURE}.csv"

with open(file_path, "w", newline="") as f:
    writer = csv.writer(f)

    count = 0
    while count < SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            # 🔑 NORMALIZATION (MOST IMPORTANT PART)
            base_x = hand.landmark[0].x
            base_y = hand.landmark[0].y
            base_z = hand.landmark[0].z

            row = []
            for lm in hand.landmark:
                row.extend([
                    lm.x - base_x,
                    lm.y - base_y,
                    lm.z - base_z
                ])

            writer.writerow(row)
            count += 1

            # Draw skeleton
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            cv2.putText(
                frame,
                f"{GESTURE} : {count}/{SAMPLES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("Collecting Gesture Data", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
            break

cap.release()
cv2.destroyAllWindows()
