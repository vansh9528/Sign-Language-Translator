import cv2
import mediapipe as mp
import csv
import os

GESTURE = "BAD"   # <-- yahan gesture ka naam change karna
SAMPLES = 200

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

os.makedirs("landmark_dataset", exist_ok=True)
file_path = f"landmark_dataset/{GESTURE}.csv"

with open(file_path, "a", newline="") as f:
    writer = csv.writer(f)

    count = 0
    while count < SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            row = []
            for lm in landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            writer.writerow(row)
            count += 1
            print(f"{GESTURE} sample {count}")

        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
