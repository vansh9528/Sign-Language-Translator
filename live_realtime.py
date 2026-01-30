import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

last_prediction = ""
stable_count = 0
STABLE_FRAMES = 8  # smoothing (no flicker)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        features = []
        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])

        probs = model.predict_proba([features])[0]
        pred = model.classes_[np.argmax(probs)]
        confidence = np.max(probs)

        if pred == last_prediction:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= STABLE_FRAMES and confidence > 0.6:
            display_text = pred
        else:
            display_text = "..."

        last_prediction = pred

        cv2.putText(
            frame,
            f"Gesture: {display_text}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("REAL-TIME ASL PROJECT", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
