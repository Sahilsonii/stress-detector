import cv2
import os
from pathlib import Path
from datetime import datetime

# Configuration
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMAGES_PER_EMOTION = 100
BASE_DIR = Path('dataset')
TRAIN_DIR = BASE_DIR / 'train'
VAL_DIR = BASE_DIR / 'validation'
TRAIN_SPLIT = 0.8  # 80% train, 20% validation

def setup_folders(person_name):
    """Create folder structure for dataset"""
    for emotion in EMOTIONS:
        (TRAIN_DIR / emotion).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / emotion).mkdir(parents=True, exist_ok=True)
    print(f"✓ Folders created for {person_name}")

def capture_emotion_images(emotion, person_name, cap):
    """Capture images for a specific emotion"""
    print(f"\n{'='*60}")
    print(f"NEXT EMOTION: {emotion.upper()}")
    print(f"{'='*60}")
    print(f"Prepare to make {emotion} expression")
    print(f"Target: {IMAGES_PER_EMOTION} images will be captured automatically")
    print("Press 'q' during capture to skip this emotion")
    print(f"{'='*60}\n")
    
    input(f">>> Press ENTER to start capturing {emotion.upper()} <<<")
    
    count = 0
    train_count = int(IMAGES_PER_EMOTION * TRAIN_SPLIT)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    frame_skip = 0
    
    while count < IMAGES_PER_EMOTION:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display info
        remaining = IMAGES_PER_EMOTION - count
        cv2.putText(frame, f"Emotion: {emotion.upper()}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Captured: {count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Remaining: {remaining}", (10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        
        # Progress bar
        bar_width = 400
        bar_height = 30
        bar_x, bar_y = 10, 150
        progress = int((count / IMAGES_PER_EMOTION) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress, bar_y + bar_height), (0, 255, 0), -1)
        cv2.putText(frame, f"{int((count/IMAGES_PER_EMOTION)*100)}%", (bar_x + bar_width + 10, bar_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, "Q: Skip Emotion", (10, frame.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Dataset Capture', frame)
        
        # Auto-capture every 3 frames when face detected
        if len(faces) > 0 and frame_skip % 3 == 0:
            folder = TRAIN_DIR if count < train_count else VAL_DIR
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{person_name}_{emotion}_{count+1}_{timestamp}.jpg"
            filepath = folder / emotion / filename
            
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            cv2.imwrite(str(filepath), face)
            
            count += 1
            print(f"✓ Auto-saved: {filename} ({count}/{IMAGES_PER_EMOTION})")
        
        frame_skip += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Skipped {emotion}")
            return False
    
    cv2.destroyAllWindows()
    print(f"\n✓ Completed {emotion.upper()}: {count} images captured")
    print(f"{'='*60}")
    return True

def main():
    print("="*60)
    print("FACIAL EMOTION DATASET CAPTURE TOOL")
    print("="*60)
    print(f"\nEmotions to capture: {', '.join(EMOTIONS)}")
    print(f"Images per emotion: {IMAGES_PER_EMOTION}")
    print(f"Total images: {len(EMOTIONS) * IMAGES_PER_EMOTION}")
    print(f"Split: {int(TRAIN_SPLIT*100)}% train, {100-int(TRAIN_SPLIT*100)}% validation\n")
    
    person_name = input("Enter your name: ").strip().replace(' ', '_')
    if not person_name:
        person_name = f"person_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    setup_folders(person_name)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "="*60)
    print("INSTRUCTIONS")
    print("="*60)
    print("1. Position your face in front of camera")
    print("2. Make the emotion expression shown on screen")
    print("3. Images will be captured AUTOMATICALLY")
    print("4. Press Q to skip current emotion")
    print("="*60)
    
    for i, emotion in enumerate(EMOTIONS, 1):
        print(f"\n[{i}/{len(EMOTIONS)}] Next emotion: {emotion.upper()}")
        capture_emotion_images(emotion, person_name, cap)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("DATASET CAPTURE COMPLETED!")
    print("="*60)
    print(f"Dataset saved to: {BASE_DIR}")
    print(f"Train folder: {TRAIN_DIR}")
    print(f"Validation folder: {VAL_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
