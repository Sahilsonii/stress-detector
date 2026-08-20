# combined_multimodal_app.py
import cv2
import joblib
import time
import numpy as np
import threading
from tkinter import *
from tensorflow.keras.models import load_model
from collections import deque
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import pickle
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# =========================================
# CONFIG
# =========================================
BASE_DIR = Path(__file__).resolve().parent

# Use the full saved model, not just weights
FACIAL_MODEL_PATH = BASE_DIR / "results" / "ResNet50V2" / "saved_model"
TYPING_MODEL_PATH = BASE_DIR / "results" / "Keystroke" / "random_forest_model.pkl"
FUSION_MODEL_PATH = BASE_DIR / "fusion_model.pkl"   # optional trained fusion classifier
IMG_SIZE = (224, 224)  # ResNet expects 224x224

SMOOTHING_WINDOW = 8
UPDATE_INTERVAL = 10000  # 10 seconds
DEFAULT_ALPHA = 0.6

print("\n📁 Facial Model Path:", FACIAL_MODEL_PATH)
print("📁 Typing Model Path:", TYPING_MODEL_PATH)
print("📁 Fusion Model Path:", FUSION_MODEL_PATH, "\n")

# =========================================
# HELPER: LOAD FUSION MODEL
# =========================================
def load_fusion_model(path=FUSION_MODEL_PATH):
    if path.exists():
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            print("🔁 Loaded fusion model from:", path)
            return model
        except Exception as e:
            print("⚠️ Failed to load fusion model:", e)
            return None
    print("ℹ️ No fusion model found; using weighted fusion.")
    return None


# =========================================
# FACIAL DETECTION THREAD
# =========================================
class FacialStressDetector(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        print("🧠 Loading facial stress model...")

        import tensorflow as tf
        
        # Load the SavedModel format (TensorFlow directory)
        try:
            self.model = tf.keras.models.load_model(str(FACIAL_MODEL_PATH))
            print("✅ Loaded model from SavedModel format")
        except Exception as e:
            print(f"⚠️ Failed to load SavedModel: {e}")
            print("Trying to load .h5 weights with architecture...")
            
            # Build ResNet50V2 architecture
            from tensorflow.keras.applications import ResNet50V2
            from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
            from tensorflow.keras.models import Model
            
            base_model = ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights=None)
            x = base_model.output
            x = GlobalAveragePooling2D(name='gap')(x)
            x = Dropout(0.5, name='dropout_1')(x)
            x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_1')(x)
            x = BatchNormalization(name='bn_1')(x)
            x = Dropout(0.3, name='dropout_2')(x)
            x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_2')(x)
            x = BatchNormalization(name='bn_2')(x)
            x = Dropout(0.2, name='dropout_3')(x)
            outputs = Dense(1, activation='sigmoid', name='output')(x)
            
            self.model = Model(inputs=base_model.input, outputs=outputs)
            
            # Load weights
            weights_path = BASE_DIR / "results" / "ResNet50V2" / "final_model_weights.h5"
            self.model.load_weights(str(weights_path))
            print(f"✅ Loaded weights from {weights_path}")
        
        print("📌 Model loaded successfully!")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.pred_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.last_probs = np.array([0.5, 0.5])
        self.current_status = "No Face"
        self.running = True

    def preprocess_face(self, face):
        face = cv2.resize(face, IMG_SIZE)
        face = face / 255.0
        return np.expand_dims(face, axis=0)

    # Removed - smoothing now done in run() method

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return

        print("📸 Camera started — Press 'q' to close camera")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT
        )


            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                pred = self.model.predict(self.preprocess_face(face_roi), verbose=0)[0]

                # Model outputs single probability for stressed class
                stressed_prob = float(pred[0])
                not_stressed_prob = 1.0 - stressed_prob
                self.last_probs = np.array([not_stressed_prob, stressed_prob])
                
                # Smooth the probabilities
                self.pred_buffer.append(self.last_probs)
                if len(self.pred_buffer) > 0:
                    self.last_probs = np.mean(np.array(self.pred_buffer), axis=0)
                
                label_idx = int(np.argmax(self.last_probs))
                self.current_status = "Stressed" if label_idx == 1 else "Not Stressed"

            else:
                 self.current_status = "No Face Detected"


            cv2.imshow("Facial Stress Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# =========================================
# MULTIMODAL UI
# =========================================
class CombinedStressUI:
    def __init__(self, alpha=DEFAULT_ALPHA):
        print("⌨️ Loading typing model...")
        if TYPING_MODEL_PATH.exists():
            self.typing_model = joblib.load(TYPING_MODEL_PATH)
            print("🔍 DEBUG: Typing model loaded:", type(self.typing_model))
        else:
            self.typing_model = None
            print(f"⚠️ No typing model at {TYPING_MODEL_PATH} -- run train_keystroke_rf.py first. Typing modality disabled; facial-only fusion.")

        self.fusion_model = load_fusion_model()

        self.facial_detector = FacialStressDetector()
        self.facial_detector.start()

        self.alpha = alpha
        self.text_buffer = ""
        self.backspaces = 0
        self._key_down_times = {}
        self._key_events = []  # (down_ts, up_ts) pairs, for hold/flight/duration features

        # UI
        self.root = Tk()
        self.root.title("Multimodal Stress Detection")
        self.root.geometry("900x650")
        self.root.configure(bg="#121212")

        Label(self.root, text="Real-Time Multimodal Stress Detection",
              font=("Arial", 20, "bold"), bg="#121212", fg="#00e0ff").pack(pady=20)

        self.typing_box = Text(self.root, height=8, width=80, bg="#1e1e2e",
                               fg="white", insertbackground="white",
                               font=("Arial", 14), wrap=WORD)
        self.typing_box.pack(pady=15)
        self.typing_box.bind("<KeyPress>", self.on_keypress)
        self.typing_box.bind("<KeyRelease>", self.on_keyrelease)

        self.typing_label = Label(self.root, font=("Arial", 14, "bold"),
                                  bg="#121212", fg="#bbbbbb")
        self.typing_label.pack()

        self.facial_label = Label(self.root, font=("Arial", 14, "bold"),
                                  bg="#121212", fg="#bbbbbb")
        self.facial_label.pack()

        self.final_label = Label(self.root, font=("Arial", 18, "bold"),
                                 bg="#121212", fg="#00e0ff")
        self.final_label.pack(pady=20)

        self.root.after(UPDATE_INTERVAL, self.update_stress)

    # -------------------------------------

    def on_keypress(self, event):
        char = event.char
        if event.keysym == "BackSpace":
            self.backspaces += 1
            if len(self.text_buffer) > 0:
                self.text_buffer = self.text_buffer[:-1]
        elif char and char.isprintable():
            self.text_buffer += char
        self._key_down_times[event.keysym] = time.time()

    def on_keyrelease(self, event):
        down_ts = self._key_down_times.pop(event.keysym, None)
        if down_ts is not None:
            self._key_events.append((down_ts, time.time()))

    def calculate_error_rate(self):
        words = len(self.text_buffer.split())
        err = float(self.backspaces) / (words + 1)
        print(f"🧮 DEBUG: error_rate={err:.4f}, words={words}, backspaces={self.backspaces}")
        return err

    def extract_typing_features(self):
        """Same hold/flight/duration schema train_keystroke_rf.py was trained
        on -- see keystroke_features.py. Returns None if too few keys typed
        since the last update to compute meaningful statistics."""
        events = sorted(self._key_events, key=lambda e: e[0])
        if len(events) < 2:
            return None
        holds = np.array([up - down for down, up in events])
        flights = np.array([events[i][0] - events[i - 1][1] for i in range(1, len(events))])
        dd = np.array([events[i][0] - events[i - 1][0] for i in range(1, len(events))])
        return np.array([[holds.mean(), holds.std(), flights.mean(), flights.std(), float(dd.sum())]])

    # -------------------------------------

    def update_stress(self):
        error_rate = self.calculate_error_rate()

        typing_p = None
        if self.typing_model is not None:
            X = self.extract_typing_features()
            if X is not None:
                typing_p = float(self.typing_model.predict_proba(X)[0][1])
        if typing_p is None:
            typing_p = 0.0  # no typing signal yet this cycle -- facial modality carries the estimate

        facial_p = float(self.facial_detector.last_probs[1])

        print(f"📊 DEBUG: Facial_p={facial_p:.4f}, Typing_p={typing_p:.4f}")

        # fusion
        if self.fusion_model:
            try:
                Xf = np.array([[facial_p, typing_p, error_rate]])
                fusion_p = float(self.fusion_model.predict_proba(Xf)[0][1])
            except Exception as e:
                print("⚠️ Fusion error, fallback:", e)
                fusion_p = self.alpha * facial_p + (1 - self.alpha) * typing_p
        else:
            fusion_p = self.alpha * facial_p + (1 - self.alpha) * typing_p

        print(f"🔮 DEBUG FUSION RESULT = {fusion_p:.4f}")

        # update UI with both probabilities
        not_stressed_facial = float(self.facial_detector.last_probs[0])
        self.typing_label.config(text=f"Typing: Stressed {typing_p*100:.1f}% | Not Stressed {(1-typing_p)*100:.1f}%",
                                 fg="#ff4444" if typing_p > 0.5 else "#44ff44")
        self.facial_label.config(text=f"Facial: Stressed {facial_p*100:.1f}% | Not Stressed {not_stressed_facial*100:.1f}%",
                                 fg="#ff4444" if facial_p > 0.5 else "#44ff44")
        self.final_label.config(text=f"Final: Stressed {fusion_p*100:.1f}% | Not Stressed {(1-fusion_p)*100:.1f}%",
                                fg="#ff3333" if fusion_p > 0.5 else "#33ff99")

        # reset buffer
        self.text_buffer = ""
        self.backspaces = 0
        self.typing_box.delete("1.0", END)

        self.root.after(UPDATE_INTERVAL, self.update_stress)

    def run(self):
        print("🚀 UI Running...")
        self.root.mainloop()
        self.facial_detector.running = False


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    CombinedStressUI().run()
