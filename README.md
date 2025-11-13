# Multimodal AI Framework for Non-Invasive Stress Detection  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-orange)](https://www.tensorflow.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A privacy-preserving, software-only system for real-time stress detection using facial expressions and keystroke dynamics.**

---

## 📋 Overview
This project introduces a multimodal AI framework for **non-invasive, real-time stress detection** in workplace environments. Unlike wearable-based systems, this method uses:

- **Facial Expression Analysis**  
  CNN models: MobileNetV2, EfficientNetB0, ResNet50V2  
  Trained on **53,571 augmented images**

- **Keystroke Dynamics**  
  Random Forest classifier analyzing typing patterns, error rates, and behavioral signals.

### Why This Matters
- ✅ No wearable sensors required  
- ✅ Privacy-preserving (local-only processing)  
- ✅ Real-time inference (~28 FPS CPU)  
- ✅ High accuracy across modalities  
- ✅ Cost-effective and scalable  

---

## ✨ Key Features

### Facial Expression Modality
- Transfer learning from ImageNet  
- Evaluated models: **MobileNetV2 (best)**, EfficientNetB0, ResNet50V2  
- Data augmentation: rotation, shift, shear, zoom, flip  
- 2-phase training:  
  - 30 epochs (trainable head)  
  - 10 epochs fine-tuning  

### Keystroke Dynamics
- Dataset: CMU Keystroke Benchmark (51 participants)  
- Features: error_rate, backspace_count, total_words  
- Random Forest with GroupKFold cross-validation  
- Optimal classification threshold: **0.65**

---

## 📊 Dataset

### Facial Expression Dataset
- **Total Images:** 53,571  
- **Training:** 40,096  
- **Validation:** 13,475  
- **Binary Labels:**  
  - **Stressed:** angry, disgust, fear, sad, neutral  
  - **Not Stressed:** happy, surprise  

### Keystroke Dataset
- CMU Keystroke Dynamics Benchmark  
- ~4,800 trials  
- 51 participants  
- Group-aware train/val/test split  

---

## 🏗️ Model Architectures

### ⭐ MobileNetV2 (Best Performing)
**Layers:**
- MobileNetV2 base (ImageNet weights)  
- GAP → Dense(256) + BN + Dropout(0.5)  
- Dense(128) + BN + Dropout(0.3)  
- Output Softmax (2 classes)

**Performance:**
- Accuracy: **84.79%**  
- Precision: 0.9446  
- Recall: 0.8361  
- F1-Score: 0.8870  
- AUC-ROC: 0.9357  
- Model Size: 14 MB  
- Speed: **28 FPS (CPU)**

---

### ⭐ Random Forest (Keystroke)
```
RandomForestClassifier(
    n_estimators=60,
    max_depth=10,
    min_samples_split=5,
    max_features='sqrt',
    class_weight='balanced'
)
```

**Performance:**
- Accuracy: **84.56% ± 1.68%**  
- Precision: 0.8734  
- Recall: 0.8134  
- F1-Score: 0.8426  

**Feature Importance:**
1. error_rate — 52.34%  
2. backspace_count — 31.56%  
3. total_words — 16.10%  

---

## 📈 Performance Comparison

| Model           | Modality | Accuracy | Precision | Recall | F1 | AUC |
|----------------|----------|----------|-----------|--------|-----|-----|
| MobileNetV2     | Facial   | 84.79%   | 0.9446    | 0.8361 | 0.8870 | 0.9357 |
| EfficientNetB0  | Facial   | 83.56%   | 0.8645    | 0.8648 | 0.8646 | 0.9245 |
| ResNet50V2      | Facial   | 82.34%   | 0.8411    | 0.8407 | 0.8409 | 0.9134 |
| Random Forest   | Keystroke| 84.56%   | 0.8734    | 0.8134 | 0.8426 | — |
| Decision Tree   | Keystroke| 76.45%   | 0.7834    | 0.7456 | 0.7642 | — |

---

## 🚀 Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/multimodal-stress-detection.git
cd multimodal-stress-detection
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Requirements
```
pip install -r requirements.txt
```

### requirements.txt
```
tensorflow==2.10.1
keras==2.10.0
opencv-python==4.6.0.66
numpy==1.23.3
scikit-learn==1.1.2
scipy==1.9.1
matplotlib==3.6.0
pandas==1.5.1
pillow==9.2.0
```

---

## ⚡ Quick Start

### Train Facial Model
```bash
python train_mobilenet.py --epochs 40 --batch_size 8
```

### Train Keystroke Model
```bash
python train_keystroke_rf.py --dataset data/cmu_keystroke.csv
```

### Run Real-Time Detection
```bash
python webcam_stress_detector.py --model results/MobileNetV2/best_model.h5
```

---

## 📁 Project Structure
```
multimodal-stress-detection/
├── data/
│   ├── original_dataset/
│   ├── balanced_train/
│   ├── balanced_validation/
│   └── cmu_keystroke.csv
├── models/
│   ├── mobilenet_model.py
│   ├── efficientnet_model.py
│   ├── resnet_model.py
│   └── keystroke_rf.py
├── results/
│   ├── MobileNetV2/
│   ├── EfficientNetB0/
│   ├── ResNet50V2/
│   └── FINAL_MODEL_COMPARISON_REPORT.pdf
├── scripts/
│   ├── train_mobilenet.py
│   ├── train_efficientnet.py
│   ├── train_resnet.py
│   ├── train_keystroke_rf.py
│   └── webcam_stress_detector.py
├── stress_monitoring/
├── docs/
├── requirements.txt
└── README.md
```

---

## 💻 Usage Examples

### Facial Stress Detection
```python
from models.mobilenet_model import load_model
import cv2

model = load_model('results/MobileNetV2/best_model.h5')
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
face = preprocess_face(frame)
prediction = model.predict(face)

print("Stress:", prediction)
```

### Keystroke Analysis
```python
from models.keystroke_rf import KeystrokeStressDetector

detector = KeystrokeStressDetector()
detector.load_model('results/keystroke_rf.pkl')

sample = {"backspace_count": 5, "total_words": 20, "error_rate": 0.25}
label, confidence = detector.predict(sample)
print(label, confidence)
```

### Fusion Model
```python
from models.multimodal_fusion import MultimodalStressDetector

detector = MultimodalStressDetector(
    facial_model='results/MobileNetV2/best_model.h5',
    keystroke_model='results/keystroke_rf.pkl'
)

score = detector.fuse_predictions(0.85, 0.78, fusion_method='late')
print(score)
```

---

## 🤝 Contributing
We welcome contributions!

### Areas to contribute:
- Fusion strategies (early/intermediate/late)  
- LSTM/GRU temporal modeling  
- Domain adaptation  
- Fairness-aware training  
- Additional behavioral modalities  

---

## ⚖️ Ethical Considerations
- **Informed consent required**  
- **No surveillance** — designed for wellness, not employee scoring  
- **Local processing only**  
- **Full data deletion supported**  
- **Human oversight mandatory**  

**Current known limitations:**  
- Dataset demographic bias  
- Proxy label assumptions  
- Temporal behavior modeling limitations  
- Real-world domain shift  

---


**⭐ If this project helps your work, please consider giving it a star! ⭐**  

