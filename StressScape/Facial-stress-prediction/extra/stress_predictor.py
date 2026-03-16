import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Load YOLOv8 model for face detection
face_detector = YOLO('yolov8n-face.pt')  # or use 'yolov8n.pt' for general detection

# Load stress detection model (assuming you have a YOLOv8 model trained for stress detection)
stress_detector = YOLO('yolov8n-stress.pt')  # replace with your trained model path

# Labels for stress detection
labels = ["Not Stressed", "Stressed"]

# Start webcam
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection using YOLOv8
    face_results = face_detector(frame, conf=0.5)[0]
    
    # Process each detected face
    for face_box in face_results.boxes.data:
        x1, y1, x2, y2, conf, _ = face_box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Extract face ROI
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue
            
        # Stress detection on the face
        stress_results = stress_detector(face_roi)[0]
        
        # Get prediction
        if len(stress_results.boxes.data) > 0:
            stress_conf = stress_results.boxes.conf[0]
            stress_cls = int(stress_results.boxes.cls[0])
            label = f"{labels[stress_cls]} ({stress_conf:.2f})"
        else:
            label = "Unknown"

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Stress Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()