# Modern Multimodal Stress Monitor with Global Keyboard Tracking
import cv2
import joblib
import time
import numpy as np
import threading
from tkinter import *
from tkinter import ttk
from tensorflow.keras.models import load_model
from collections import deque
from pathlib import Path
from pynput import keyboard
import pickle
import warnings
warnings.filterwarnings("ignore")

# CONFIG
BASE_DIR = Path(__file__).resolve().parent
FACIAL_MODEL_PATH = BASE_DIR / "results" / "ResNet50V2" / "saved_model"
TYPING_MODEL_PATH = BASE_DIR.parent.parent / "error_rate" / "StressScape" / "stress_detector.pkl"
IMG_SIZE = (224, 224)  # ResNet expects 224x224
SMOOTHING_WINDOW = 8
UPDATE_INTERVAL = 5000  # 5 seconds

class GlobalKeyboardMonitor:
    def __init__(self):
        self.total_keys = 0
        self.backspaces = 0
        self.words_typed = 0
        self.text_buffer = []
        self.running = True
        self.listener = None
        
    def on_press(self, key):
        if not self.running:
            return False
        
        try:
            if hasattr(key, 'char') and key.char and key.char.isprintable():
                self.total_keys += 1
                self.text_buffer.append(key.char)
                if key.char == ' ':
                    self.words_typed += 1
            elif key == keyboard.Key.backspace:
                self.backspaces += 1
                if self.text_buffer:
                    self.text_buffer.pop()
        except:
            pass
    
    def start(self):
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        print("⌨️ Global keyboard monitoring started")
    
    def stop(self):
        self.running = False
        if self.listener:
            self.listener.stop()
    
    def get_error_rate(self):
        if self.total_keys == 0:
            return 0.0
        return self.backspaces / max(self.total_keys, 1)
    
    def reset(self):
        self.total_keys = 0
        self.backspaces = 0
        self.words_typed = 0
        self.text_buffer = []

class FacialStressDetector(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        print("🧠 Loading ResNet model...")
        import tensorflow as tf
        
        # Load SavedModel format
        self.model = tf.keras.models.load_model(str(FACIAL_MODEL_PATH))
        print("✅ ResNet model loaded from SavedModel format")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.pred_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.last_probs = np.array([0.5, 0.5])
        self.current_status = "No Face"
        self.confidence = 0.0
        self.running = True
        print("✅ Facial model loaded!")
    
    def preprocess_face(self, face):
        face = cv2.resize(face, IMG_SIZE)
        return np.expand_dims(face / 255.0, axis=0)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not accessible")
            return
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                pred = self.model.predict(self.preprocess_face(face_roi), verbose=0)[0]
                # Model outputs single probability for stressed class
                stressed_prob = float(pred[0])
                not_stressed_prob = 1.0 - stressed_prob
                probs = np.array([not_stressed_prob, stressed_prob])
                
                self.pred_buffer.append(probs)
                smooth = np.mean(np.array(self.pred_buffer), axis=0)
                self.last_probs = smooth
                label_idx = int(np.argmax(smooth))
                self.confidence = smooth[label_idx] * 100
                self.current_status = "Stressed" if label_idx == 1 else "Not Stressed"
                
                color = (0, 0, 255) if label_idx == 1 else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, f"{self.current_status} ({self.confidence:.1f}%)", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                self.current_status = "No Face"
            
            cv2.imshow("Facial Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def stop(self):
        self.running = False

class ModernStressUI:
    def __init__(self):
        print("⌨️ Loading typing model...")
        self.typing_model = joblib.load(TYPING_MODEL_PATH)
        
        self.keyboard_monitor = GlobalKeyboardMonitor()
        self.keyboard_monitor.start()
        
        self.facial_detector = FacialStressDetector()
        self.facial_detector.start()
        
        self.root = Tk()
        self.root.title("🧠 Real-Time Multimodal Stress Monitor")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0a0e27")
        
        self.setup_ui()
        self.root.after(UPDATE_INTERVAL, self.update_display)
    
    def setup_ui(self):
        # Header
        header = Frame(self.root, bg="#1a1f3a", height=100)
        header.pack(fill=X, pady=(0, 20))
        
        Label(header, text="🧠 MULTIMODAL STRESS DETECTION SYSTEM", 
              font=("Segoe UI", 28, "bold"), bg="#1a1f3a", fg="#00d4ff").pack(pady=20)
        
        # Main container
        main = Frame(self.root, bg="#0a0e27")
        main.pack(fill=BOTH, expand=True, padx=30, pady=10)
        
        # Left panel - Metrics
        left = Frame(main, bg="#0a0e27")
        left.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 15))
        
        # Facial Stress Card
        self.create_card(left, "👤 FACIAL STRESS", "facial")
        
        # Typing Stress Card
        self.create_card(left, "⌨️ TYPING STRESS", "typing")
        
        # Keyboard Stats Card
        self.create_stats_card(left)
        
        # Right panel - Final Result
        right = Frame(main, bg="#0a0e27")
        right.pack(side=RIGHT, fill=BOTH, expand=True)
        
        self.create_final_card(right)
    
    def create_card(self, parent, title, card_type):
        card = Frame(parent, bg="#1a1f3a", relief=RAISED, bd=2)
        card.pack(fill=X, pady=10)
        
        Label(card, text=title, font=("Segoe UI", 16, "bold"), 
              bg="#1a1f3a", fg="#00d4ff").pack(pady=15)
        
        status_frame = Frame(card, bg="#1a1f3a")
        status_frame.pack(pady=10)
        
        if card_type == "facial":
            self.facial_status = Label(status_frame, text="Waiting...", 
                                      font=("Segoe UI", 24, "bold"), bg="#1a1f3a", fg="#888")
            self.facial_status.pack()
            
            self.facial_conf = Label(card, text="Confidence: --", 
                                    font=("Segoe UI", 14), bg="#1a1f3a", fg="#aaa")
            self.facial_conf.pack(pady=10)
            
            self.facial_prob = ttk.Progressbar(card, length=300, mode='determinate')
            self.facial_prob.pack(pady=10)
        
        else:  # typing
            self.typing_status = Label(status_frame, text="Monitoring...", 
                                      font=("Segoe UI", 24, "bold"), bg="#1a1f3a", fg="#888")
            self.typing_status.pack()
            
            self.typing_error = Label(card, text="Error Rate: --", 
                                     font=("Segoe UI", 14), bg="#1a1f3a", fg="#aaa")
            self.typing_error.pack(pady=10)
            
            self.typing_prob = ttk.Progressbar(card, length=300, mode='determinate')
            self.typing_prob.pack(pady=10)
    
    def create_stats_card(self, parent):
        card = Frame(parent, bg="#1a1f3a", relief=RAISED, bd=2)
        card.pack(fill=X, pady=10)
        
        Label(card, text="📊 KEYBOARD STATISTICS", font=("Segoe UI", 16, "bold"), 
              bg="#1a1f3a", fg="#00d4ff").pack(pady=15)
        
        stats_frame = Frame(card, bg="#1a1f3a")
        stats_frame.pack(pady=10, padx=20, fill=X)
        
        self.keys_label = Label(stats_frame, text="Total Keys: 0", 
                               font=("Segoe UI", 12), bg="#1a1f3a", fg="#fff", anchor=W)
        self.keys_label.pack(fill=X, pady=5)
        
        self.backspace_label = Label(stats_frame, text="Backspaces: 0", 
                                    font=("Segoe UI", 12), bg="#1a1f3a", fg="#fff", anchor=W)
        self.backspace_label.pack(fill=X, pady=5)
        
        self.words_label = Label(stats_frame, text="Words: 0", 
                                font=("Segoe UI", 12), bg="#1a1f3a", fg="#fff", anchor=W)
        self.words_label.pack(fill=X, pady=5)
    
    def create_final_card(self, parent):
        card = Frame(parent, bg="#1a1f3a", relief=RAISED, bd=3)
        card.pack(fill=BOTH, expand=True)
        
        Label(card, text="🎯 FINAL STRESS ASSESSMENT", font=("Segoe UI", 20, "bold"), 
              bg="#1a1f3a", fg="#00d4ff").pack(pady=30)
        
        self.final_status = Label(card, text="ANALYZING...", 
                                 font=("Segoe UI", 36, "bold"), bg="#1a1f3a", fg="#888")
        self.final_status.pack(pady=40)
        
        self.final_conf = Label(card, text="Confidence: --", 
                               font=("Segoe UI", 18), bg="#1a1f3a", fg="#aaa")
        self.final_conf.pack(pady=20)
        
        self.final_prob = ttk.Progressbar(card, length=400, mode='determinate')
        self.final_prob.pack(pady=20)
        
        Label(card, text=f"Updates every {UPDATE_INTERVAL//1000} seconds", 
              font=("Segoe UI", 10), bg="#1a1f3a", fg="#666").pack(pady=20)
    
    def update_display(self):
        # Facial
        facial_prob = float(self.facial_detector.last_probs[1])
        facial_stressed = facial_prob > 0.5
        self.facial_status.config(
            text=self.facial_detector.current_status,
            fg="#ff3333" if facial_stressed else "#33ff33"
        )
        self.facial_conf.config(text=f"Confidence: {self.facial_detector.confidence:.1f}%")
        self.facial_prob['value'] = facial_prob * 100
        
        # Typing
        error_rate = self.keyboard_monitor.get_error_rate()
        input_arr = np.array([[error_rate]])
        
        if hasattr(self.typing_model, "predict_proba"):
            typing_probs = self.typing_model.predict_proba(input_arr)[0]
            typing_prob = float(typing_probs[1])
        else:
            typing_pred = int(self.typing_model.predict(input_arr)[0])
            typing_prob = 1.0 if typing_pred == 1 else 0.0
        
        typing_stressed = typing_prob > 0.5
        self.typing_status.config(
            text="STRESSED" if typing_stressed else "NOT STRESSED",
            fg="#ff3333" if typing_stressed else "#33ff33"
        )
        self.typing_error.config(text=f"Error Rate: {error_rate:.3f}")
        self.typing_prob['value'] = typing_prob * 100
        
        # Stats
        self.keys_label.config(text=f"Total Keys: {self.keyboard_monitor.total_keys}")
        self.backspace_label.config(text=f"Backspaces: {self.keyboard_monitor.backspaces}")
        self.words_label.config(text=f"Words: {self.keyboard_monitor.words_typed}")
        
        # Final (weighted fusion: 60% facial, 40% typing)
        final_prob = 0.6 * facial_prob + 0.4 * typing_prob
        final_stressed = final_prob > 0.5
        
        self.final_status.config(
            text="🔴 STRESSED" if final_stressed else "🟢 NOT STRESSED",
            fg="#ff3333" if final_stressed else "#33ff33"
        )
        self.final_conf.config(text=f"Confidence: {final_prob*100:.1f}%")
        self.final_prob['value'] = final_prob * 100
        
        # Reset keyboard stats
        self.keyboard_monitor.reset()
        
        self.root.after(UPDATE_INTERVAL, self.update_display)
    
    def run(self):
        print("🚀 Modern UI running with global keyboard monitoring")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
    
    def on_close(self):
        self.keyboard_monitor.stop()
        self.facial_detector.stop()
        self.root.destroy()

if __name__ == "__main__":
    app = ModernStressUI()
    app.run()
