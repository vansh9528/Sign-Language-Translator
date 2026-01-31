"""
Small, clean utils module used by the project.
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



class MJPEGStream:
    """Simple MJPEG stream reader that exposes a .read() API similar to OpenCV's VideoCapture.

    It runs a background thread to fetch bytes from the HTTP MJPEG stream and decodes JPEG frames.
    """

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
            bytes_buff = b''
            self.opened = True
            for chunk in resp.iter_content(chunk_size=1024):
                if not self.running:
                    break
                if chunk:
                    bytes_buff += chunk
                    a = bytes_buff.find(b'\xff\xd8')
                    b = bytes_buff.find(b'\xff\xd9')
                    if a != -1 and b != -1 and b > a:
                        jpg = bytes_buff[a:b+2]
                        bytes_buff = bytes_buff[b+2:]
                        try:
                            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                with self.lock:
                                    self.frame = img
                        except Exception:
                            # skip invalid frame
                            pass
        except Exception as e:
            print(f"MJPEG stream reader error: {e}")
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
    """Create and start an MJPEGStream, with a small wait for it to become ready."""
    stream = MJPEGStream(url)
    stream.start()
    wait = 0
    while not stream.isOpened() and wait < retry_count:
        time.sleep(retry_delay)
        wait += 1
    if not stream.isOpened():
        raise RuntimeError(f"Failed to open MJPEG stream: {url}")
    return stream


def initialize_hand_detector():
    """Initialize MediaPipe hand detector with configured settings."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        static_image_mode=STATIC_IMAGE_MODE,
    )
    return hands, mp_hands


def initialize_camera(backend="DSHOW"):
    """
    Initialize camera with specified backend.

    Supports local webcams (by index) and HTTP MJPEG streams (ESP32).
    Tries OpenCV VideoCapture first; if that fails it falls back to the requests-based MJPEGStream.
    """
    # If configured to use ESP32 or CAMERA_INDEX is a URL, try opening the MJPEG stream
    if USE_ESP32 or (isinstance(CAMERA_INDEX, str) and CAMERA_INDEX.startswith("http")):
        url = ESP32_STREAM_URL if USE_ESP32 else CAMERA_INDEX
        print(f"Trying to open ESP32 stream: {url}")

        # Try OpenCV VideoCapture first (may work if OpenCV built with ffmpeg)
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

        # VideoCapture failed — fall back to requests-based MJPEG reader
        print("OpenCV VideoCapture failed for ESP32 stream; switching to requests-based MJPEG fallback")
        mj = create_mjpeg_stream(url, ESP32_RETRY_COUNT, ESP32_RETRY_DELAY)
        return mj

    # Otherwise open a local camera by index using the requested backend
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
    """
    Extract normalized hand landmarks (relative to wrist, scale-invariant).

    Args:
        hand_landmarks: MediaPipe hand landmarks object

    Returns:
        list: Flattened list of 21 landmarks × 3 coordinates (x, y, z) = 63 features
    """
    # Step 1: Use wrist (landmark 0) as origin
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    base_z = hand_landmarks.landmark[0].z

    # Step 2: Compute relative coordinates and find max distance
    rel_points = []
    for lm in hand_landmarks.landmark:
        rel_x = lm.x - base_x
        rel_y = lm.y - base_y
        rel_z = lm.z - base_z
        rel_points.append((rel_x, rel_y, rel_z))

    # Step 3: Scale by max distance (makes it scale-invariant)
    max_dist = max((abs(x) + abs(y) + abs(z)) for (x, y, z) in rel_points)

    # Avoid division by zero
    if max_dist < 0.001:
        max_dist = 1.0

    # Step 4: Normalize and flatten
    features = []
    for (x, y, z) in rel_points:
        features.extend([x / max_dist, y / max_dist, z / max_dist])

    return features


def draw_landmarks_on_frame(frame, hand_landmarks, mp_hands):
    """
    Draw hand skeleton on frame.
    """
    mp_draw = mp.solutions.drawing_utils
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame


def add_text_to_frame(frame, text, position=(30, 40), font_scale=1, color=(0, 255, 0), thickness=2):
    """
    Add text to frame.
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame
"""
Utility functions for Sign Language Translator
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import requests
import io
import time
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
    ESP32_RETRY_DELAY
)


def initialize_hand_detector():
    """Initialize MediaPipe hand detector with configured settings."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        static_image_mode=STATIC_IMAGE_MODE
    )
    return hands, mp_hands


def initialize_camera(backend="DSHOW"):
    """
    Initialize camera with specified backend.
    
    Args:
        backend (str): Camera backend ("DSHOW", "V4L2", "AVFOUNDATION")
    
    Returns:
        cv2.VideoCapture: Camera object
    """
    # If configured to use ESP32 or CAMERA_INDEX is a URL, try opening the MJPEG stream
    if USE_ESP32 or (isinstance(CAMERA_INDEX, str) and CAMERA_INDEX.startswith("http")):
        url = ESP32_STREAM_URL if USE_ESP32 else CAMERA_INDEX
        print(f"Trying to open ESP32 stream: {url}")
        # Try OpenCV VideoCapture first (may work if OpenCV built with ffmpeg)
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

        # VideoCapture failed — fall back to requests-based MJPEG reader
        print("OpenCV VideoCapture failed for ESP32 stream; switching to requests-based MJPEG fallback")
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
class MJPEGStream:
    """Simple MJPEG stream reader that exposes a .read() API similar to OpenCV's VideoCapture.

    It runs a background thread to fetch bytes from the HTTP MJPEG stream and decodes JPEG frames.
    """
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
            bytes_buff = b''
            self.opened = True
            for chunk in resp.iter_content(chunk_size=1024):
                if not self.running:
                    break
                if chunk:
                    bytes_buff += chunk
                    a = bytes_buff.find(b'\xff\xd8')
                    b = bytes_buff.find(b'\xff\xd9')
                    if a != -1 and b != -1 and b > a:
                        jpg = bytes_buff[a:b+2]
                        bytes_buff = bytes_buff[b+2:]
                        try:
                            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                with self.lock:
                                    self.frame = img
                        except Exception:
                            # skip invalid frame
                            pass
        except Exception as e:
            print(f"MJPEG stream reader error: {e}")
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
    """Create and start an MJPEGStream, with a small wait for it to become ready."""
    stream = MJPEGStream(url)
    stream.start()
    wait = 0
    while not stream.isOpened() and wait < retry_count:
        time.sleep(retry_delay)
        wait += 1
    if not stream.isOpened():
        raise RuntimeError(f"Failed to open MJPEG stream: {url}")
    return stream
                self.thread.start()

            def _reader(self):
                try:
                    resp = requests.get(self.url, stream=True, timeout=5)
                    if resp.status_code != 200:
                        print(f"MJPEG stream returned status {resp.status_code}")
                        self.opened = False
                        return
                    boundary = None
                    content_type = resp.headers.get('Content-Type', '')
                    if 'boundary=' in content_type:
                        boundary = content_type.split('boundary=')[-1]

                    bytes_buff = b''
                    self.opened = True
                    for chunk in resp.iter_content(chunk_size=1024):
                        if not self.running:
                            break
                        if chunk:
                            bytes_buff += chunk
                            a = bytes_buff.find(b'\xff\xd8')
                            b = bytes_buff.find(b'\xff\xd9')
                            if a != -1 and b != -1 and b > a:
                                jpg = bytes_buff[a:b+2]
                                bytes_buff = bytes_buff[b+2:]
                                try:
                                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                                    if img is not None:
                                        with self.lock:
                                            self.frame = img
                                except Exception as e:
                                    # skip invalid frame
                                    pass
                except Exception as e:
                    print(f"MJPEG stream reader error: {e}")
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

        mj = MJPEGStream(url)
        mj.start()
        # Wait a short moment for the reader to populate a frame
        wait = 0
        while not mj.isOpened() and wait < ESP32_RETRY_COUNT:
            time.sleep(ESP32_RETRY_DELAY)
            wait += 1

        if not mj.isOpened():
            raise RuntimeError(f"Failed to open ESP32 stream via VideoCapture and MJPEG fallback: {url}")

        return mj

    # Otherwise open a local camera by index using the requested backend
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
    """
    Extract normalized hand landmarks (relative to wrist, scale-invariant).
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
    
    Returns:
        list: Flattened list of 21 landmarks × 3 coordinates (x, y, z) = 63 features
    """
    # Step 1: Use wrist (landmark 0) as origin
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    base_z = hand_landmarks.landmark[0].z

    # Step 2: Compute relative coordinates and find max distance
    rel_points = []
    for lm in hand_landmarks.landmark:
        rel_x = lm.x - base_x
        rel_y = lm.y - base_y
        rel_z = lm.z - base_z
        rel_points.append((rel_x, rel_y, rel_z))

    # Step 3: Scale by max distance (makes it scale-invariant)
    max_dist = max(
        (abs(x) + abs(y) + abs(z)) for (x, y, z) in rel_points
    )
    
    # Avoid division by zero
    if max_dist < 0.001:
        max_dist = 1.0

    # Step 4: Normalize and flatten
    features = []
    for (x, y, z) in rel_points:
        features.extend([
            x / max_dist,
            y / max_dist,
            z / max_dist
        ])
    
    return features


def draw_landmarks_on_frame(frame, hand_landmarks, mp_hands):
    """
    Draw hand skeleton on frame.
    
    Args:
        frame: OpenCV frame
        hand_landmarks: MediaPipe hand landmarks
        mp_hands: MediaPipe hands module
    
    Returns:
        frame: Modified frame with drawn landmarks
    """
    mp_draw = mp.solutions.drawing_utils
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame


def add_text_to_frame(frame, text, position=(30, 40), font_scale=1, 
                      color=(0, 255, 0), thickness=2):
    """
    Add text to frame.
    
    Args:
        frame: OpenCV frame
        text: Text to add
        position: (x, y) position
        font_scale: Font size
        color: BGR color tuple
        thickness: Text thickness
    
    Returns:
        frame: Modified frame with text
    """
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    return frame
