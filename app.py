from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import threading
import os
from gtts import gTTS
import pygame
from ultralytics import YOLO
import speech_recognition as sr
import numpy as np
import sounddevice as sd
import queue
import pytesseract
from pytesseract import Output
import pyttsx3
import sys
import re
import copy
import itertools
import pandas as pd
import string
import io
from scipy.io.wavfile import write
from pydub import AudioSegment 
import subprocess
import mediapipe as mp
from tensorflow.keras.models import load_model
import base64

try:
    import pythoncom
except ImportError:
    pythoncom = None

try:
    import win32com.client
except ImportError:
    win32com = None

try:
    import winsound
except ImportError:
    winsound = None

app = Flask(__name__)

# ==================== OBJECT DETECTION CONFIGURATION ====================
model = YOLO('yolov8s.pt')
IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.6
MAX_FPS = 15
SPEAK_DELAY = 5

pygame.mixer.init()
AUDIO_FOLDER = "audio_cache"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Object Detection Global Variables
object_camera_active = False
detection_enabled = True
last_spoken = {}
last_detection_time = 0
detected_objects = set()
object_cap = None
frame_lock = threading.Lock()

# ==================== TEXT DETECTION CONFIGURATION ====================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Text Detection Global Variables
text_camera_active = False
last_spoken_text = ""
MIN_TEXT_LENGTH = 1
THROTTLING_DELAY_SECONDS = 3
MIN_WORD_CONFIDENCE = 60
MIN_ALPHA_CHARACTERS = 4
text_cap = None

# ==================== SIGN LANGUAGE DETECTION CONFIGURATION ====================
MODEL_FILE = "sign_model.h5"
LABEL_MAP = "label_map.npy"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load sign language model safely
try:
    sign_model = load_model(MODEL_FILE)
    sign_labels = np.load(LABEL_MAP, allow_pickle=True)
    print(f"✅ Sign Language Model loaded with {len(sign_labels)} classes: {', '.join(sign_labels)}")
except Exception as e:
    print(f"❌ Error loading sign language model: {e}")
    sign_model, sign_labels = None, []

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# ==================== CHARACTER DETECTION CONFIGURATION ====================
CHARACTER_MODEL_FILE = "model.h5"

# Character Detection Global Variables
character_camera_active = False
character_cap = None
character_current_prediction = "None"
character_current_confidence = 0.0

# Load character detection model
try:
    character_model = load_model(CHARACTER_MODEL_FILE)
    character_alphabet = ['1','2','3','4','5','6','7','8','9']
    character_alphabet += list(string.ascii_uppercase)
    print(f"✅ Character Detection Model loaded with {len(character_alphabet)} classes")
except Exception as e:
    print(f"❌ Error loading character detection model: {e}")
    character_model = None
    character_alphabet = []

# ==================== TTS WORKER FOR TEXT DETECTION ====================
class TTSWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.text_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.engine = None
        self.is_running = False
        self.com_initialized = False
        self.sapi_voice = None
        self.last_backend_used = "pyttsx3"
        self.powershell_available = sys.platform.startswith("win")
        self.beep_enabled = winsound is not None
        self.enforce_output_device = None

    def _initialize_engine(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            voices = engine.getProperty('voices')

            voice_id_to_use = None
            voice_name_to_use = "Default"

            if voices:
                for v in voices:
                    try:
                        name = (v.name or "").lower()
                    except Exception:
                        name = ""
                    if any(k in name for k in ("zira", "david", "microsoft", "anna", "net")):
                        voice_id_to_use = v.id
                        voice_name_to_use = v.name
                        break

                if voice_id_to_use is None:
                    for v in voices:
                        if getattr(v, 'id', None):
                            voice_id_to_use = v.id
                            voice_name_to_use = getattr(v, 'name', 'unknown')
                            break

                if voice_id_to_use:
                    try:
                        engine.setProperty('voice', voice_id_to_use)
                        print(f"TTS Engine initialized. Using voice: {voice_name_to_use}")
                    except Exception as se:
                        print(f"Warning: could not set voice: {se}")
            return engine
        except Exception as e:
            print(f"TTS initialization error: {e}")
            return None

    def _initialize_windows_sapi_voice(self):
        if not sys.platform.startswith("win") or win32com is None:
            return None
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            voice.Volume = 100
            voice.Rate = -1
            print("Windows SAPI voice ready.")
            return voice
        except Exception as e:
            print(f"SAPI init failed: {e}")
            return None

    def _speak_with_powershell(self, text):
        if not self.powershell_available:
            return False
        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak([Console]::In.ReadToEnd())"
        )
        try:
            process = subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", ps_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True
            )
            process.communicate(text, timeout=15)
            return process.returncode == 0
        except Exception:
            return False

    def run(self):
        if sys.platform.startswith("win"):
            if pythoncom is not None and not self.com_initialized:
                try:
                    pythoncom.CoInitialize()
                    self.com_initialized = True
                except Exception as e:
                    print(f"COM init failed: {e}")

        if sys.platform.startswith("win"):
            self.sapi_voice = self._initialize_windows_sapi_voice()

        self.engine = self._initialize_engine()
        if self.engine is None and self.sapi_voice is None:
            print("No TTS backend available.")
            self.is_running = False
            return

        if self.engine is None and self.sapi_voice is not None:
            self.last_backend_used = "sapi"

        self.is_running = True
        print("TTS Worker ready.")

        while not self.stop_event.is_set():
            try:
                text, rate = self.text_queue.get(timeout=0.1)
                backend_used = None

                if self.beep_enabled:
                    try:
                        winsound.Beep(1200, 200)
                    except RuntimeError:
                        self.beep_enabled = False

                if self.sapi_voice is not None and sys.platform.startswith("win"):
                    try:
                        sapi_rate = max(-10, min(10, int(round((rate - 150) / 10.0))))
                        self.sapi_voice.Rate = sapi_rate
                        self.sapi_voice.Volume = 100
                        self.sapi_voice.Speak(text)
                        backend_used = "sapi"
                    except Exception:
                        self.sapi_voice = None

                if backend_used is None and self.engine is not None:
                    try:
                        self.engine.setProperty('rate', rate)
                        self.engine.stop()
                        self.engine.say(text)
                        self.engine.runAndWait()
                        backend_used = "pyttsx3"
                    except Exception:
                        self.engine = None

                if backend_used is None:
                    if self._speak_with_powershell(text):
                        backend_used = "powershell"

                self.text_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS Worker Error: {e}")

        if self.engine:
            self.engine.stop()
        self.sapi_voice = None
        self.is_running = False
        if self.com_initialized and pythoncom is not None:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def stop(self):
        self.stop_event.set()

    def put(self, text, rate=150):
        with self.text_queue.mutex:
            self.text_queue.queue.clear()
        self.text_queue.put((text, rate))

tts_worker = TTSWorker()

# ==================== TEXT DETECTION HELPER FUNCTIONS ====================
def speak_text_detection(text, rate=150):
    clean_text = ' '.join(text.split())
    if not tts_worker.is_running and not getattr(tts_worker, 'is_alive', lambda: False)():
        try:
            tts_worker.start()
        except RuntimeError:
            pass

    if clean_text.strip():
        if tts_worker.is_running:
            tts_worker.put(clean_text, rate)
        else:
            print("TTS worker not ready.")

def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text).lower().strip()
    return ' '.join(text.split())

def is_new_text(new_text):
    global last_spoken_text
    new_norm = normalize_text(new_text)
    last_norm = normalize_text(last_spoken_text)

    if len(new_norm) < MIN_TEXT_LENGTH:
        return False

    import difflib
    ratio = difflib.SequenceMatcher(None, new_norm, last_norm).ratio()
    return ratio < 0.85

# ==================== OBJECT DETECTION FUNCTIONS ====================
def speak_object_detection(text):
    print(f"🔊 Speaking: {text}")
    safe_text_hash = str(hash(text))
    audio_path = os.path.join(AUDIO_FOLDER, f"{safe_text_hash}.mp3")

    try:
        if not os.path.exists(audio_path):
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(audio_path)

        while pygame.mixer.music.get_busy():
            time.sleep(0.01)

        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"❌ Audio error: {e}")

def describe_scene(objects):
    if not objects:
        return "No objects detected in your surroundings"

    objects_list = list(objects)
    objects_count = {}
    for obj in objects_list:
        objects_count[obj] = objects_count.get(obj, 0) + 1

    description_parts = []
    for obj, count in objects_count.items():
        if count == 1:
            description_parts.append(f"a {obj}")
        else:
            description_parts.append(f"{count} {obj}s")

    if len(description_parts) == 1:
        return f"I can see {description_parts[0]} in front of you."
    elif len(description_parts) == 2:
        return f"I see {description_parts[0]} and {description_parts[1]}."
    else:
        last_item = description_parts.pop()
        return f"I see several objects: {', '.join(description_parts)}, and {last_item}."

# ==================== SIGN LANGUAGE DETECTION FUNCTIONS ====================
def extract_landmarks(img):
    """Extract hand landmarks from image"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, 
                       min_detection_confidence=0.5) as hands:
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return None, None
        features = []
        for hand_landmarks in results.multi_hand_landmarks:
            coords = []
            for p in hand_landmarks.landmark:
                coords.extend([p.x, p.y, p.z])
            wrist_x, wrist_y, wrist_z = coords[0], coords[1], coords[2]
            norm_coords = []
            for i in range(0, len(coords), 3):
                norm_coords.extend([
                    coords[i] - wrist_x,
                    coords[i + 1] - wrist_y,
                    coords[i + 2] - wrist_z
                ])
            features.extend(norm_coords)
        if len(results.multi_hand_landmarks) == 1:
            features.extend([0.0] * 63)
        return np.array(features, dtype=np.float32), results

def create_hand_only_image(img, results):
    """
    Create a processed image showing only hands with background removed
    - Extract hand regions
    - Apply grayscale
    - Remove background
    - Enhance hand visibility
    """
    h, w = img.shape[:2]
    
    # Create mask for hands
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box of hand
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            
            x_min = max(0, min(x_coords) - 30)
            x_max = min(w, max(x_coords) + 30)
            y_min = max(0, min(y_coords) - 30)
            y_max = min(h, max(y_coords) + 30)
            
            # Create hand region mask
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
    
    # Apply mask to image
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    
    # Convert back to BGR for consistent display
    img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # Enhance contrast
    img_gray_bgr = cv2.convertScaleAbs(img_gray_bgr, alpha=1.3, beta=10)
    
    # Draw landmarks
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_gray_bgr, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    return img_gray_bgr

def draw_landmarks(img, results):
    """Draw hand landmarks on image with colored overlay"""
    img_copy = img.copy()
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_copy, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    return img_copy

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

# ==================== UNIFIED VOICE LISTENER ====================
def unified_voice_listener():
    global object_camera_active, text_camera_active, detection_enabled

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000
    microphone = sr.Microphone()

    threading.Thread(target=speak_object_detection, args=("Voice control is now active.",), daemon=True).start()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.5)

    print("🎤 Unified voice recognition ready.")

    while True:
        try:
            with microphone as source:
                audio = recognizer.listen(source, timeout=4, phrase_time_limit=4)

            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"🎤 Heard: '{command}'")

                if "close" in command:
                    if object_camera_active or text_camera_active:
                        object_camera_active = False
                        text_camera_active = False
                        threading.Thread(target=speak_object_detection, args=("Camera closed.",), daemon=True).start()
                        print("ℹ️ All cameras deactivated by close command")
                
                elif "open text camera" in command or ("text" in command and "open" in command):
                    if not text_camera_active:
                        text_camera_active = True
                        object_camera_active = False
                        guidance_message = "Text camera opened. Please place the document about 8 inches from the lens."
                        threading.Thread(target=speak_text_detection, args=(guidance_message,)).start()
                        print("✅ Text camera activated")
                
                elif "open camera" in command and "text" not in command:
                    if not object_camera_active:
                        object_camera_active = True
                        text_camera_active = False
                        threading.Thread(target=speak_object_detection, args=("Object camera started.",), daemon=True).start()
                        print("✅ Object camera activated")

                elif "toggle detection" in command:
                    detection_enabled = not detection_enabled
                    status = "enabled" if detection_enabled else "disabled"
                    threading.Thread(target=speak_object_detection, args=(f"Detection {status}.",), daemon=True).start()

            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"❌ Speech API error: {e}")

        except sr.WaitTimeoutError:
            pass
        except Exception as e:
            print(f"❌ Voice listener error: {e}")
            time.sleep(1)
# ==================== CHARACTER DETECTION HELPER FUNCTIONS ====================
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    
    def normalize_(n):
        return n / max_value
    
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
    return temp_landmark_list

# ==================== FRAME GENERATION ====================
def generate_frames():
    global object_camera_active, text_camera_active, detection_enabled
    global last_detection_time, detected_objects, object_cap, text_cap
    global last_spoken_text

    if not tts_worker.is_running:
        tts_worker.start()
        time.sleep(0.5)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)

    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera Error - Check Connection", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    last_ocr_time = time.time()

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if object_camera_active:
                if detection_enabled:
                    results = model(frame, imgsz=IMG_SIZE, verbose=False)[0]
                    current_detected = set()

                    for result in results.boxes:
                        conf = float(result.conf[0])
                        if conf >= CONFIDENCE_THRESHOLD:
                            cls_id = int(result.cls[0])
                            label = model.names[cls_id]
                            current_detected.add(label)

                            x1, y1, x2, y2 = map(int, result.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    detected_objects = current_detected

                    current_time = time.time()
                    if detected_objects and (current_time - last_detection_time) > SPEAK_DELAY:
                        description = describe_scene(detected_objects)
                        threading.Thread(target=speak_object_detection, args=(description,), daemon=True).start()
                        last_detection_time = current_time
                else:
                    cv2.putText(frame, "Detection DISABLED", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                status_text = "OBJECT DETECTION - " + ("ON" if detection_enabled else "OFF")
                status_color = (0, 255, 0) if detection_enabled else (0, 165, 255)
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            elif text_camera_active:
                h, w, _ = frame.shape
                current_time = time.time()

                x_start, y_start = int(w * 0.2), int(h * 0.2)
                x_end, y_end = int(w * 0.8), int(h * 0.8)

                roi_frame = frame[y_start:y_end, x_start:x_end]
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

                if tts_worker.text_queue.empty() and (current_time - last_ocr_time) > THROTTLING_DELAY_SECONDS:
                    last_ocr_time = current_time

                    text = ""
                    avg_confidence = 0.0
                    try:
                        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                        thresholded_roi = cv2.adaptiveThreshold(
                            blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY, 11, 2
                        )
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        clean_thresholded_roi = cv2.morphologyEx(thresholded_roi, cv2.MORPH_CLOSE, kernel)

                        ocr_data = pytesseract.image_to_data(
                            clean_thresholded_roi, lang='eng', 
                            config='--psm 6', output_type=Output.DICT
                        )

                        words = []
                        confidences = []

                        for word, conf in zip(ocr_data.get('text', []), ocr_data.get('conf', [])):
                            cleaned_word = word.strip()
                            try:
                                conf_val = float(conf)
                            except (TypeError, ValueError):
                                conf_val = -1.0

                            if not cleaned_word or conf_val < MIN_WORD_CONFIDENCE:
                                continue

                            words.append(cleaned_word)
                            confidences.append(conf_val)

                        if words and confidences:
                            avg_confidence = sum(confidences) / len(confidences)
                            raw_text = " ".join(words)
                            text = re.sub(r'[^\x00-\x7F]+', ' ', raw_text)

                            alpha_chars = sum(1 for ch in text if ch.isalpha())
                            if alpha_chars < MIN_ALPHA_CHARACTERS:
                                text = ""

                    except Exception as e:
                        print(f"OCR Error: {e}")

                    should_speak = text.strip() and avg_confidence > 0

                    if should_speak and is_new_text(text):
                        last_spoken_text = text
                        threading.Thread(target=speak_text_detection, args=(text,)).start()

                status_text = "TEXT DETECTION - Center text in green box"
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            else:
                cv2.putText(frame, "Vision Assistant - Cameras OFF", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Say 'Open Camera' or 'Open Text Camera'", (50, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        elapsed = time.time() - start_time
        sleep_time = max(1.0 / MAX_FPS - elapsed, 0)
        time.sleep(sleep_time)

    cap.release()
# ==================== CHARACTER DETECTION FRAME GENERATION ====================
def generate_character_frames():
    """Generate frames for character detection"""
    global character_camera_active, character_current_prediction, character_current_confidence
    global character_cap

    if character_model is None:
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Character model not loaded", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    if character_cap is None or not character_cap.isOpened():
        character_cap = cv2.VideoCapture(0)
        character_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        character_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            success, image = character_cap.read()
            
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for selfie-view
            image = cv2.flip(image, 1)
            
            if character_camera_active:
                # Process the image
                image.flags.writeable = False
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                debug_image = copy.deepcopy(image)

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                          results.multi_handedness):
                        # Calculate landmarks
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        
                        # Draw landmarks
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        
                        # Prepare data for prediction
                        df = pd.DataFrame(pre_processed_landmark_list).transpose()
                        
                        # Predict
                        predictions = character_model.predict(df, verbose=0)
                        predicted_classes = np.argmax(predictions, axis=1)
                        character_current_prediction = character_alphabet[predicted_classes[0]]
                        character_current_confidence = float(np.max(predictions) * 100)
                        
                        # Display prediction on image
                        cv2.putText(image, character_current_prediction, 
                                  (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                  3, (0, 255, 0), 4)
                        
                        cv2.putText(image, f"Confidence: {character_current_confidence:.1f}%", 
                                  (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.8, (255, 255, 255), 2)
                else:
                    character_current_prediction = "No hand detected"
                    character_current_confidence = 0.0
                    cv2.putText(image, "Show your hand", (50, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            else:
                cv2.putText(image, "Character Detection - Ready", (50, 200),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, "Click 'Start Detection' to begin", (50, 250),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Encode and yield frame
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign-language')
def sign_language():
    return render_template('sign_language.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera/<camera_type>', methods=['POST'])
def toggle_camera(camera_type):
    global object_camera_active, text_camera_active
    
    if camera_type == 'object':
        object_camera_active = not object_camera_active
        if object_camera_active:
            text_camera_active = False
            threading.Thread(target=speak_object_detection, args=("Object camera started.",), daemon=True).start()
        else:
            threading.Thread(target=speak_object_detection, args=("Object camera stopped.",), daemon=True).start()
    elif camera_type == 'text':
        text_camera_active = not text_camera_active
        if text_camera_active:
            object_camera_active = False
            threading.Thread(target=speak_text_detection, args=("Text camera started.",)).start()
        else:
            threading.Thread(target=speak_text_detection, args=("Text camera stopped.",)).start()
    
    return jsonify({'status': 'success'})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_enabled
    detection_enabled = not detection_enabled
    status = "enabled" if detection_enabled else "disabled"
    threading.Thread(target=speak_object_detection, args=(f"Detection {status}",), daemon=True).start()
    return jsonify({'status': 'success', 'message': f'Detection {status}'})

@app.route('/speak_input_text', methods=['POST'])
def speak_input_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'status': 'error', 'message': 'No text provided'}), 400
        
        threading.Thread(target=speak_text_detection, args=(text,)).start()
        
        return jsonify({'status': 'success', 'message': 'Speaking text'})
    except Exception as e:
        print(f"Error in speak_input_text: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/detect_sign', methods=['POST'])
def detect_sign():
    if sign_model is None:
        return jsonify({'error': 'Sign language model not loaded properly.'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    try:
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract landmarks
        features, results = extract_landmarks(img)
        if features is None:
            return jsonify({'error': 'No hands detected. Please show your hands clearly.'}), 400
        
        # Make prediction
        features = features.reshape(1, -1)
        predictions = sign_model.predict(features, verbose=0)[0]
        top_3_idx = np.argsort(predictions)[-3:][::-1]
        top_predictions = [
            {'label': sign_labels[idx], 'confidence': float(predictions[idx] * 100)}
            for idx in top_3_idx
        ]
        
        # Create processed image (hands only, grayscale)
        img_processed = create_hand_only_image(img, results)
        
        # For comparison, also create colored landmark version
        img_with_landmarks = draw_landmarks(img, results)
        
        return jsonify({
            'prediction': top_predictions[0]['label'],
            'confidence': top_predictions[0]['confidence'],
            'top_predictions': top_predictions,
            'original_image': image_to_base64(img_with_landmarks),
            'detected_image': image_to_base64(img_processed)
        })
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/status')
def status():
    return jsonify({
        'object_camera_active': object_camera_active,
        'text_camera_active': text_camera_active,
        'detection_enabled': detection_enabled,
        'detected_objects': list(detected_objects)
    })

fs = 44100
@app.route('/speech')
def speech_page():
    return render_template('record.html', text=None, recording=False)

@app.route('/start_record', methods=['POST'])
def start_record():
    duration = 5
    print("Recording started...")
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")

    rec_int16 = np.int16(rec * 32767)
    memfile = io.BytesIO()
    write(memfile, fs, rec_int16)
    memfile.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(memfile) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Sorry, I could not understand the audio."
        except sr.RequestError:
            text = "Speech recognition service unavailable."

    return render_template('record.html', text=text, recording=False)

@app.route('/stop_record', methods=['POST'])
def stop_record():
    return ("", 204)

@app.route('/upload', methods=['POST'])
def upload_audio():
    file = request.files['audio']
    if file:
        recognizer = sr.Recognizer()

        input_bytes = io.BytesIO(file.read())
        try:
            with sr.AudioFile(input_bytes) as source:
                audio_data = recognizer.record(source)
        except Exception:
            input_bytes.seek(0)
            audio = AudioSegment.from_file(input_bytes)
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            with sr.AudioFile(wav_buffer) as source:
                audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Sorry, I could not understand the audio."
        except sr.RequestError:
            text = "Speech recognition service unavailable."

        return render_template('record.html', text=text, recording=False)
    return render_template('record.html', text="No file uploaded.", recording=False)
try:
    from cvzone.HandTrackingModule import HandDetector
    from cvzone.ClassificationModule import Classifier
    cvzone_available = True
except ImportError:
    print("⚠️ cvzone not installed. Real-time detection will not work.")
    cvzone_available = False
@app.route('/character-detection')
def character_detection():
    return render_template('character_detection.html')

@app.route('/character_feed')
def character_feed():
    return Response(generate_character_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_character_detection', methods=['POST'])
def start_character_detection():
    global character_camera_active
    
    if character_model is None:
        return jsonify({
            'status': 'error',
            'message': 'Character detection model not loaded'
        })
    
    character_camera_active = True
    return jsonify({
        'status': 'success',
        'message': 'Character detection started'
    })

@app.route('/stop_character_detection', methods=['POST'])
def stop_character_detection():
    global character_camera_active
    character_camera_active = False
    return jsonify({
        'status': 'success',
        'message': 'Character detection stopped'
    })

@app.route('/character_status')
def character_status():
    return jsonify({
        'active': character_camera_active,
        'prediction': character_current_prediction,
        'confidence': character_current_confidence
    })

import math

# ==================== REAL-TIME SIGN LANGUAGE CONFIGURATION ====================
REALTIME_MODEL_PATH = "Model/keras_model.h5"
REALTIME_LABELS_PATH = "Model/labels.txt"
REALTIME_LABELS = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

# Real-time detection global variables
realtime_camera_active = False
realtime_detector = None
realtime_classifier = None
realtime_cap = None
realtime_offset = 20
realtime_imgSize = 300
realtime_current_gesture = "None"
realtime_current_confidence = 0.0
realtime_frame_count = 0

# Initialize real-time detector and classifier
if cvzone_available:
    try:
        realtime_detector = HandDetector(maxHands=1)
        if os.path.exists(REALTIME_MODEL_PATH) and os.path.exists(REALTIME_LABELS_PATH):
            realtime_classifier = Classifier(REALTIME_MODEL_PATH, REALTIME_LABELS_PATH)
            print(f"✅ Real-time sign detection model loaded")
        else:
            print(f"⚠️ Real-time model files not found at {REALTIME_MODEL_PATH}")
            realtime_classifier = None
    except Exception as e:
        print(f"❌ Error initializing real-time detection: {e}")
        realtime_detector = None
        realtime_classifier = None

# ==================== REAL-TIME SIGN DETECTION FUNCTIONS ====================
def generate_realtime_sign_frames():
    """Generate frames for real-time sign language detection"""
    global realtime_camera_active, realtime_current_gesture, realtime_current_confidence
    global realtime_frame_count, realtime_cap

    if not cvzone_available or realtime_detector is None or realtime_classifier is None:
        # Return error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Real-time detection not available", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(error_frame, "Install cvzone: pip install cvzone", (50, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    if realtime_cap is None or not realtime_cap.isOpened():
        realtime_cap = cv2.VideoCapture(0)
        realtime_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        realtime_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, img = realtime_cap.read()
        if not success:
            break

        imgOutput = img.copy()

        if realtime_camera_active:
            try:
                hands, img = realtime_detector.findHands(img)
                
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    # Create white background image
                    imgWhite = np.ones((realtime_imgSize, realtime_imgSize, 3), np.uint8) * 255

                    # Crop hand region
                    y1 = max(0, y - realtime_offset)
                    y2 = min(img.shape[0], y + h + realtime_offset)
                    x1 = max(0, x - realtime_offset)
                    x2 = min(img.shape[1], x + w + realtime_offset)

                    imgCrop = img[y1:y2, x1:x2]

                    if imgCrop.size > 0:
                        imgCropShape = imgCrop.shape
                        aspectRatio = h / w

                        if aspectRatio > 1:
                            k = realtime_imgSize / h
                            wCal = math.ceil(k * w)
                            if wCal > 0 and wCal <= realtime_imgSize:
                                imgResize = cv2.resize(imgCrop, (wCal, realtime_imgSize))
                                wGap = math.ceil((realtime_imgSize - wCal) / 2)
                                imgWhite[:, wGap:wCal + wGap] = imgResize
                                prediction, index = realtime_classifier.getPrediction(imgWhite, draw=False)
                        else:
                            k = realtime_imgSize / w
                            hCal = math.ceil(k * h)
                            if hCal > 0 and hCal <= realtime_imgSize:
                                imgResize = cv2.resize(imgCrop, (realtime_imgSize, hCal))
                                hGap = math.ceil((realtime_imgSize - hCal) / 2)
                                imgWhite[hGap:hCal + hGap, :] = imgResize
                                prediction, index = realtime_classifier.getPrediction(imgWhite, draw=False)

                        # Update current gesture and confidence
                        if index < len(REALTIME_LABELS):
                            realtime_current_gesture = REALTIME_LABELS[index]
                            realtime_current_confidence = float(np.max(prediction))
                        else:
                            realtime_current_gesture = "Unknown"
                            realtime_current_confidence = 0.0

                        # Draw bounding box and label on output
                        cv2.rectangle(imgOutput, (x - realtime_offset, y - realtime_offset - 70),
                                    (x - realtime_offset + 400, y - realtime_offset + 60 - 50),
                                    (0, 255, 0), cv2.FILLED)

                        cv2.putText(imgOutput, realtime_current_gesture, (x, y - 30),
                                  cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                        
                        cv2.rectangle(imgOutput, (x - realtime_offset, y - realtime_offset),
                                    (x + w + realtime_offset, y + h + realtime_offset),
                                    (0, 255, 0), 4)

                        realtime_frame_count += 1
                else:
                    realtime_current_gesture = "No hand detected"
                    realtime_current_confidence = 0.0

            except Exception as e:
                print(f"Error in real-time detection: {e}")
                realtime_current_gesture = "Error"
                realtime_current_confidence = 0.0
        else:
            # Camera active but detection not started
            cv2.putText(imgOutput, "Real-time Detection - Ready", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(imgOutput, "Click 'Start Detection' to begin", (50, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.033)  # ~30 FPS

# ==================== REAL-TIME FLASK ROUTES ====================
@app.route('/realtime-sign')
def realtime_sign():
    """Real-time sign language detection page"""
    return render_template('realtime_sign.html')

@app.route('/realtime_sign_feed')
def realtime_sign_feed():
    """Video feed for real-time sign detection"""
    return Response(generate_realtime_sign_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_realtime_sign', methods=['POST'])
def start_realtime_sign():
    """Start real-time sign detection"""
    global realtime_camera_active, realtime_frame_count
    
    if not cvzone_available or realtime_detector is None or realtime_classifier is None:
        return jsonify({
            'status': 'error',
            'message': 'Real-time detection not available. Please install cvzone.'
        })
    
    realtime_camera_active = True
    realtime_frame_count = 0
    return jsonify({
        'status': 'success',
        'message': 'Real-time detection started'
    })

@app.route('/stop_realtime_sign', methods=['POST'])
def stop_realtime_sign():
    """Stop real-time sign detection"""
    global realtime_camera_active
    realtime_camera_active = False
    return jsonify({
        'status': 'success',
        'message': 'Real-time detection stopped'
    })

@app.route('/realtime_sign_status')
def realtime_sign_status():
    """Get current real-time detection status"""
    return jsonify({
        'active': realtime_camera_active,
        'gesture': realtime_current_gesture,
        'confidence': realtime_current_confidence,
        'frame_count': realtime_frame_count
    })

if __name__ == '__main__':
    voice_thread = threading.Thread(target=unified_voice_listener, daemon=True)
    voice_thread.start()

    print("\n" + "="*60)
    print("🚀 AI Vision Assistant Pro Starting...")
    print("="*60)
    print("🎤 Voice commands:")
    print("   - 'Open Camera' → Object Detection")
    print("   - 'Open Text Camera' → Text Detection")
    print("   - 'Close' → Close any active camera")
    print("   - 'Toggle Detection' → Enable/Disable object detection")
    print("="*60)
    print("🌐 Open your browser and go to:")
    print("   http://localhost:5001")
    print("   or")
    print("   http://127.0.0.1:5001")
    print("="*60)
    print("\nStarting Flask server...\n")

    try:
        app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\nTroubleshooting:")
        print("1. Check if port 5001 is already in use")
        print("2. Try running with: python app.py")
        print("3. Make sure all dependencies are installed")
