import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import time
from collections import deque
from pathlib import Path
from datetime import datetime
import csv

BASE_DIR = Path(__file__).parent
IMG_SIZE = (224, 224)
SMOOTHING_WINDOW = 10
SCREENSHOT_INTERVAL = 60

def build_model(model_type='MobileNetV2'):
    """Rebuild model architecture"""
    if model_type == 'MobileNetV2':
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    elif model_type == 'EfficientNetB0':
        base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    else:
        base_model = ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    
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
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

class StressDetector:
    def __init__(self, model_path, model_type, user_name):
        print("Loading stress detection model...")
        try:
            self.model = load_model(model_path)
            print(f"Model loaded from: {model_path}")
        except:
            print(f"Rebuilding {model_type} architecture and loading weights...")
            self.model = build_model(model_type)
            self.model.load_weights(str(model_path))
            print(f"Weights loaded from: {model_path}")
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.class_names = ['Not Stressed', 'Stressed']
        self.colors = {'Not Stressed': (0, 255, 0), 'Stressed': (0, 0, 255)}
        
        self.user_dir = Path('stress_monitoring') / user_name
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.user_dir / 'stress_log.csv'
        self.init_csv()
        
        print("Model loaded successfully!")
        print(f"Screenshots will be saved to: {self.user_dir}")
    
    def preprocess_face(self, face_img):
        face_resized = cv2.resize(face_img, IMG_SIZE)
        face_normalized = face_resized / 255.0
        return np.expand_dims(face_normalized, axis=0)
    
    def get_smoothed_prediction(self, prediction):
        self.prediction_buffer.append(prediction)
        return np.mean(self.prediction_buffer, axis=0)
    
    def init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Status', 'Confidence', 'Screenshot'])
    
    def log_to_csv(self, status, confidence, screenshot_name):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), status, f"{confidence:.2f}%", screenshot_name])
    
    def detect_stress(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        stress_status = None
        confidence = 0
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            face_input = self.preprocess_face(face_roi)
            prediction = self.model.predict(face_input, verbose=0)[0]
            
            if len(prediction.shape) == 0 or prediction.shape[0] == 1:
                stress_prob = float(prediction) if len(prediction.shape) == 0 else float(prediction[0])
                not_stress_prob = 1.0 - stress_prob
                smoothed_pred = self.get_smoothed_prediction([not_stress_prob, stress_prob])
            else:
                smoothed_pred = self.get_smoothed_prediction(prediction)
            
            if smoothed_pred[1] > 0.5:
                class_idx = 1
                confidence = smoothed_pred[1] * 100
            else:
                class_idx = 0
                confidence = smoothed_pred[0] * 100
            
            class_name = self.class_names[class_idx]
            stress_status = class_name
            color = self.colors[class_name]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            label = f"{class_name}: {confidence:.1f}%"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y-35), (x+text_width+10, y), color, -1)
            cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            stress_prob = smoothed_pred[1] * 100
            not_stress_prob = smoothed_pred[0] * 100
            prob_text = f"Stressed: {stress_prob:.1f}% | Not Stressed: {not_stress_prob:.1f}%"
            cv2.putText(frame, prob_text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, len(faces) > 0, stress_status, confidence
    
    def run(self):
        print("\n" + "="*70)
        print("AUTOMATED STRESS MONITORING SYSTEM")
        print("="*70)
        print("\n⚠️  NOTICE: Screenshots will be captured automatically every 60 seconds")
        print("📁 Screenshots and logs will be saved to:", self.user_dir)
        print("📊 Only ONE person will be monitored at a time")
        print("\nPress 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        fps_start_time = time.time()
        last_screenshot_time = time.time()
        fps_counter = 0
        fps = 0
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame, face_detected, stress_status, confidence = self.detect_stress(frame)
            
            current_time = time.time()
            if current_time - last_screenshot_time >= SCREENSHOT_INTERVAL and face_detected:
                screenshot_count += 1
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"screenshot_{screenshot_count}_{timestamp}.jpg"
                filepath = self.user_dir / filename
                cv2.imwrite(str(filepath), processed_frame)
                self.log_to_csv(stress_status, confidence, filename)
                print(f"✓ Auto-captured: {filename} | Status: {stress_status}")
                last_screenshot_time = current_time
            
            fps_counter += 1
            if current_time - fps_start_time > 1:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = current_time
            
            info_text = f"FPS: {fps} | Face: {'Detected' if face_detected else 'Not Detected'}"
            cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            time_until_next = int(SCREENSHOT_INTERVAL - (current_time - last_screenshot_time))
            countdown_text = f"Next capture in: {time_until_next}s" if face_detected else "Waiting for face..."
            cv2.putText(processed_frame, countdown_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(processed_frame, "Press 'q' to quit", 
                       (10, processed_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Stress Monitoring', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*70)
        print(f"Session completed! Total screenshots: {screenshot_count}")
        print(f"Data saved to: {self.user_dir}")
        print(f"Log file: {self.csv_path}")
        print("="*70)

def main():
    # Prioritize ResNet50V2 model
    MODEL_PATHS = [
        BASE_DIR / 'results' / 'ResNet50V2' / 'final_model_weights.h5',
        BASE_DIR / 'results' / 'ResNet50V2' / 'saved_model',
        BASE_DIR / 'results' / 'EfficientNetB0' / 'final_model_weights.h5',
        BASE_DIR / 'results' / 'EfficientNetB0' / 'saved_model',
        BASE_DIR / 'results' / 'MobileNetV2' / 'final_model_weights.h5',
        BASE_DIR / 'results' / 'MobileNetV2' / 'saved_model',
        BASE_DIR / 'best_stress_detector.h5',
        BASE_DIR / 'final_stress_detector.h5'
    ]
    
    try:
        print("="*70)
        print("STRESS MONITORING SYSTEM")
        print("="*70)
        user_name = input("\nEnter your name: ").strip()
        if not user_name:
            user_name = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = None
        for path in MODEL_PATHS:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            print("\n❌ Error: No trained model found!")
            print("\nPlease train a model first using one of:")
            print("  python train_mobilenet.py")
            print("  python train_efficientnet.py")
            print("  python train_resnet.py")
            return
        
        model_type = 'MobileNetV2'
        if 'EfficientNet' in str(model_path):
            model_type = 'EfficientNetB0'
        elif 'ResNet' in str(model_path):
            model_type = 'ResNet50V2'
        
        detector = StressDetector(model_path, model_type, user_name)
        detector.run()
        
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
