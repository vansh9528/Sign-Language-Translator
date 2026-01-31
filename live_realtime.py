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
STABLE_FRAMES = 8

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    display_text = "..."

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # 🔑 SAME NORMALIZATION AS TRAINING
        base_x = hand.landmark[0].x
        base_y = hand.landmark[0].y
        base_z = hand.landmark[0].z

        features = []
        for lm in hand.landmark:
            features.extend([
                lm.x - base_x,
                lm.y - base_y,
                lm.z - base_z
            ])

        probs = model.predict_proba([features])[0]
        pred = model.classes_[np.argmax(probs)]
        confidence = np.max(probs)

        # Stability check
        if pred == last_prediction:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= STABLE_FRAMES and confidence > 0.45:
            display_text = pred

        last_prediction = pred

        # DEBUG (temporary – optional)
        print(pred, round(confidence, 2))

    cv2.putText(
        frame,
        f"Gesture: {display_text}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("REAL-TIME SIGN LANGUAGE TRANSLATOR", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
