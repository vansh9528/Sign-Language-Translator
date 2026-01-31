"""
Clean utils module (alternate) to avoid editing the corrupted `utils.py`.
"""

import time
import threading
import cv2
import mediapipe as mp
import numpy as np
import requests

from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
    STATIC_IMAGE_MODE,
    CAMERA_INDEX,
    CAMERA_BACKEND,
    USE_ESP32,
    ESP32_STREAM_URL,
    ESP32_RETRY_COUNT,
    ESP32_RETRY_DELAY,
)


class MJPEGStream:
    def __init__(self, url):
        self.url = url
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.opened = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        try:
            resp = requests.get(self.url, stream=True, timeout=5)
            if resp.status_code != 200:
                print(f"MJPEG stream returned status {resp.status_code}")
                self.opened = False
                return
            buf = b""
            self.opened = True
            for chunk in resp.iter_content(chunk_size=1024):
                if not self.running:
                    break
                if not chunk:
                    continue
                buf += chunk
                a = buf.find(b"\xff\xd8")
                b = buf.find(b"\xff\xd9")
                if a != -1 and b != -1 and b > a:
                    jpg = buf[a : b + 2]
                    buf = buf[b + 2 :]
                    try:
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            with self.lock:
                                self.frame = img
                    except Exception:
                        pass
        except Exception as exc:
            print(f"MJPEG stream reader error: {exc}")
        finally:
            self.opened = False

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def isOpened(self):
        return self.opened

    def release(self):
        self.running = False
        try:
            if self.thread:
                self.thread.join(timeout=1)
        except Exception:
            pass


def create_mjpeg_stream(url, retry_count=5, retry_delay=1.0):
    stream = MJPEGStream(url)
    stream.start()
    waited = 0
    while not stream.isOpened() and waited < retry_count:
        time.sleep(retry_delay)
        waited += 1
    if not stream.isOpened():
        raise RuntimeError(f"Failed to open MJPEG stream: {url}")
    return stream


def initialize_hand_detector():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        static_image_mode=STATIC_IMAGE_MODE,
    )
    return hands, mp_hands


def initialize_camera(backend="DSHOW"):
    if USE_ESP32 or (isinstance(CAMERA_INDEX, str) and str(CAMERA_INDEX).startswith("http")):
        url = ESP32_STREAM_URL if USE_ESP32 else CAMERA_INDEX
        print(f"Trying to open ESP32 stream: {url}")
        cap = cv2.VideoCapture(url)
        tries = 0
        while not cap.isOpened() and tries < ESP32_RETRY_COUNT:
            print(f"ESP32 stream not open via VideoCapture, retrying ({tries+1}/{ESP32_RETRY_COUNT})...")
            time.sleep(ESP32_RETRY_DELAY)
            try:
                cap.open(url)
            except Exception:
                pass
            tries += 1
        if cap.isOpened():
            print("ESP32 stream opened with OpenCV VideoCapture")
            return cap
        print("OpenCV VideoCapture failed for ESP32 stream; switching to MJPEG fallback")
        return create_mjpeg_stream(url, ESP32_RETRY_COUNT, ESP32_RETRY_DELAY)

    backend_map = {
        "DSHOW": cv2.CAP_DSHOW,
        "V4L2": cv2.CAP_V4L2,
        "AVFOUNDATION": cv2.CAP_AVFOUNDATION,
        "MSMF": cv2.CAP_MSMF,
    }
    backend_id = backend_map.get(backend, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(CAMERA_INDEX, backend_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera with {backend} backend")
    return cap


def extract_hand_landmarks(hand_landmarks):
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    base_z = hand_landmarks.landmark[0].z
    rel_points = []
    for lm in hand_landmarks.landmark:
        rel_points.append((lm.x - base_x, lm.y - base_y, lm.z - base_z))
    max_dist = max((abs(x) + abs(y) + abs(z)) for (x, y, z) in rel_points)
    if max_dist < 0.001:
        max_dist = 1.0
    features = []
    for (x, y, z) in rel_points:
        features.extend([x / max_dist, y / max_dist, z / max_dist])
    return features


def draw_landmarks_on_frame(frame, hand_landmarks, mp_hands):
    mp_draw = mp.solutions.drawing_utils
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame


def add_text_to_frame(frame, text, position=(30, 40), font_scale=1, color=(0, 255, 0), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame
