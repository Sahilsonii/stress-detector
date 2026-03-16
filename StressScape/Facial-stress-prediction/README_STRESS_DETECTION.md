# Facial Expression Stress Detection System

Complete pipeline for creating custom datasets, training multiple stress detection models with checkpointing, and real-time webcam monitoring with automated logging.

## 📁 Project Structure

```
├── original dataset/               # Your custom captured dataset
│   ├── train/                      # Training images (7 emotions)
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── validation/                 # Validation images (7 emotions)
├── balanced_train/                 # Augmented balanced training set (generated)
├── balanced_validation/            # Augmented balanced validation set (generated)
├── results/                        # All training results organized by model
│   ├── MobileNetV2/
│   │   ├── checkpoints/            # Training checkpoints
│   │   ├── logs/                   # Training logs
│   │   ├── best_model.h5
│   │   ├── final_model.h5
│   │   ├── metrics.json
│   │   └── training_results.png
│   ├── EfficientNetB0/
│   ├── ResNet50V2/
│   └── FINAL_MODEL_COMPARISON_REPORT.pdf
├── stress_monitoring/              # Webcam monitoring logs per user
│   └── {username}/
│       ├── stress_log.csv
│       └── screenshot_*.jpg
├── logs/                           # System logs (generated)
│   └── system_*.log
├── capture_dataset.py              # Step 0: Capture custom dataset
├── data_augmentation_balancer.py   # Step 1: Balance dataset
├── train_mobilenet.py              # Step 2a: Train MobileNetV2
├── train_efficientnet.py           # Step 2b: Train EfficientNetB0
├── train_resnet.py                 # Step 2c: Train ResNet50V2
├── generate_comparison_report.py   # Step 3: Generate comparison
├── regenerate_reports_only.py      # Step 4: Regenerate reports (no training)
├── webcam_stress_detector.py       # Step 5: Automated monitoring
├── count_dataset_images.py         # Utility: Count dataset statistics
├── system_logger.py                # Utility: Comprehensive logging system
├── run_complete_pipeline.py        # Run all steps interactively
└── requirements.txt                # Dependencies
```

## 🎯 Emotion to Stress Mapping

**7 Original Emotions → 2 Stress Categories:**

- **Stressed** (Label: 1): angry, sad, fear, disgust, neutral
- **Not Stressed** (Label: 0): happy, surprise

**Rationale:**
- Negative emotions (angry, sad, fear, disgust) indicate stress
- Neutral face often indicates baseline stress or lack of positive emotion
- Positive emotions (happy, surprise) indicate absence of stress

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)
```bash
python run_complete_pipeline.py
```
Interactive script that guides you through all steps:
- **Step 0**: Capture dataset (optional)
- **Step 1**: Balance dataset with augmentation
- **Step 2a**: Train MobileNetV2
- **Step 2b**: Train EfficientNetB0
- **Step 2c**: Train ResNet50V2
- **Step 3**: Generate comparison report
- **Step 4**: Regenerate reports only (no training)
- **Step 5**: Launch webcam detector

### Option 2: Individual Steps

Run each step separately as needed:

## 📸 Step 0: Capture Custom Dataset

Create your own dataset by capturing facial expressions via webcam.

```bash
python capture_dataset.py
```

**Features:**
- Captures 100 images per emotion automatically
- 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise
- Auto face detection and cropping
- 80/20 train/validation split
- Real-time progress display with countdown
- Press ENTER to start each emotion
- Press Q to skip emotion

**Output:**
- Creates `original dataset/train/` and `original dataset/validation/` folders
- Total: 700 images (100 per emotion)
- Organized by emotion class

## ⚖️ Step 1: Balance Dataset with Augmentation

This script analyzes your dataset, finds the class with the most images, and augments all other classes to match that count.

```bash
python data_augmentation_balancer.py
```

**What it does:**
- Counts images in each emotion folder
- Identifies the maximum class count
- Generates augmented images for minority classes using:
  - Rotation (±20°)
  - Width/Height shifts (±20%)
  - Shear transformation (15%)
  - Zoom (±15%)
  - Horizontal flipping
- Creates balanced_train/ and balanced_validation/ folders

**Output:**
```
Original Training Set:
  angry       :  4153 images
  disgust     :   578 images
  fear        :  4254 images
  happy       :  5728 images  ← Maximum
  neutral     :  5141 images
  sad         :  5094 images
  surprise    :  3352 images
  TOTAL       : 28300 images

Target: 5728 images per class

Balanced Training Set:
  All emotions:  5728 images each
  TOTAL       : 40096 images

Augmented: +11796 images 
  Training Set:   +41.7%
  Validation Set: +81.3%
```
0
## 📊 Dataset Statistics

🔹 Training Set:
  angry       :  5728 images
  disgust     :  5728 images
  fear        :  5728 images
  happy       :  5728 images
  neutral     :  5728 images
  sad         :  5728 images
  surprise    :  5728 images
  TOTAL       : 40096 images

🔹 Validation Set:
  angry       :  1925 images
  disgust     :  1925 images
  fear        :  1925 images
  happy       :  1925 images
  neutral     :  1925 images
  sad         :  1925 images
  surprise    :  1925 images
  TOTAL       : 13475 images

📊 Balanced Dataset Total: 53571 images

📈 AUGMENTATION SUMMARY

✨ Augmented Images Created:
  Training Set:   11796 images
  Validation Set:  6044 images
  TOTAL:          17840 images

📊 Dataset Growth:
  Training Set:   +41.7%
  Validation Set: +81.3%

✨ Augmented Images Created:
  Training Set:   11796 images
  Validation Set:  6044 images
  TOTAL:          17840 images

📊 Dataset Growth:
  Training Set:   +41.7%
  Validation Set: +81.3%
  TOTAL:          17840 images

📊 Dataset Growth:
  Training Set:   +41.7%
  Validation Set: +81.3%
  Validation Set: +81.3%
  Overall:        +49.9%

## 🤖 Step 2: Train Models (Individual Scripts)

Each model has its own training script to prevent retraining everything if one fails.

### Train MobileNetV2
```bash
python train_mobilenet.py
```

### Train EfficientNetB0
```bash
python train_efficientnet.py
```

### Train ResNet50V2
```bash
python train_resnet.py
```

**Training Features:**
- **3 Models**: MobileNetV2, EfficientNetB0, ResNet50V2
- **Two-phase Training**:
  - Phase 1: Train classifier head (30 epochs)
  - Phase 2: Fine-tune base layers (10 epochs)
  - Total: 40 epochs per model
- **Batch Size**: 8 (optimized for memory efficiency)
- **Checkpointing**: Saves every epoch + best model based on validation accuracy
- **Resume Training**: Automatically detects and resumes from last checkpoint
- **Comprehensive Logging**: All training logs saved to `results/{model}/logs/`
- **Automatic Report Generation**: Creates training plots, confusion matrix, and PDF report

**Output Structure:**
```
results/
├── MobileNetV2/
│   ├── checkpoints/
│   │   ├── checkpoint_epoch_01_val_acc_0.8523.h5
│   │   ├── checkpoint_epoch_02_val_acc_0.8745.h5
│   │   ├── best_model.h5              # Best performing checkpoint
│   │   └── training_log.csv           # Epoch-by-epoch metrics
│   ├── logs/
│   │   └── training_20240115_143022.log  # Detailed training log
│   ├── final_model.h5                 # Final trained model
│   ├── metrics.json                   # Performance metrics
│   ├── training_results.png           # Plots (accuracy, loss, confusion matrix)
│   └── classification_report.txt      # Detailed classification metrics
├── EfficientNetB0/                    # Same structure
└── ResNet50V2/                        # Same structure
```

**Resume Training:**
If training is interrupted, simply run the script again. It will:
1. Detect existing checkpoints
2. Ask if you want to resume
3. Continue from the last saved epoch

## 📊 Step 3: Generate Comparison Report

After training all models, generate comprehensive comparison:

```bash
python generate_comparison_report.py
```

**Generates:**
- Individual model performance with training curves and confusion matrices
- Side-by-side confusion matrix comparison for all models
- Modern visualizations (bar charts, radar chart, heatmap, box plots)
- Comprehensive metrics analysis table
- Final model selection with highlighted winner and detailed rationale
- Professional PDF report with all sections

**Output:**
```
results/
├── comparison/                        # Comparison visualizations
│   ├── metrics_comparison.png
│   ├── all_confusion_matrices.png
│   ├── heatmap_comparison.png
│   └── final_decision.png
└── FINAL_MODEL_COMPARISON_REPORT.pdf  # Comprehensive PDF report
```

**Best Model Selection:**
The report automatically identifies and highlights the best performing model based on validation accuracy.

## 📄 Step 4: Regenerate Reports Only (No Training)

Regenerate reports from existing trained models without retraining:

```bash
python regenerate_reports_only.py
```

**Use this when:**
- Models are already trained
- You want to update visualizations
- You need a fresh PDF report
- No training required, just reads existing metrics.json files

## 📹 Step 5: Automated Webcam Stress Monitoring

Real-time stress detection with automated screenshot capture and CSV logging.

```bash
python webcam_stress_detector.py
```

**Features:**
- **Automated Monitoring**: Screenshots captured every 60 seconds automatically
- **Single Person Focus**: Monitors only one person at a time
- **User-Specific Folders**: Each user gets their own folder
- **CSV Logging**: Records timestamp, status, confidence, and screenshot filename
- **Real-time Display**:
  - Face detection with bounding box
  - Stress status with confidence percentage
  - Color-coded: 🟢 Green (Not Stressed) | 🔴 Red (Stressed)
  - Countdown timer to next capture
  - FPS counter
- **Prediction Smoothing**: Averages over 10 frames for stability

**Controls:**
- `q` - Quit and save session

**Output Structure:**
```
stress_monitoring/
└── {username}/
    ├── stress_log.csv              # Complete session log
    ├── screenshot_1_20240115_143022.jpg
    ├── screenshot_2_20240115_143122.jpg
    └── ...
```

**CSV Log Format:**
```csv
Timestamp,Status,Confidence,Screenshot
2024-01-15 14:30:22,Stressed,87.45%,screenshot_1_20240115_143022.jpg
2024-01-15 14:31:22,Not Stressed,92.31%,screenshot_2_20240115_143122.jpg
```

**User Notification:**
Before starting, users are notified that:
- Screenshots will be captured every 60 seconds
- Only one person will be monitored
- All data will be saved to their personal folder

## 📊 Model Architectures

### All Models Share Common Top Layers:
```
Base Model (ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu) → BatchNormalization → Dropout(0.5)
    ↓
Dense(128, relu) → BatchNormalization → Dropout(0.3)
    ↓
Dense(2, softmax) [Not Stressed, Stressed]
```

### Base Models:
1. **MobileNetV2**: Lightweight, fast inference (~3.5M params)
2. **EfficientNetB0**: Balanced efficiency and accuracy (~5.3M params)
3. **ResNet50V2**: Deep architecture, high accuracy (~25M params)

## 📈 Expected Performance

### Training:
- **Validation Accuracy**: 85-95% (depends on dataset quality)
- **Training Time**: 
  - MobileNetV2: ~2-3 hours (CPU) / ~30-45 min (GPU)
  - EfficientNetB0: ~3-4 hours (CPU) / ~45-60 min (GPU)
  - ResNet50V2: ~4-6 hours (CPU) / ~60-90 min (GPU)
- **Batch Size**: 8
- **Total Epochs**: 40 (30 initial + 10 fine-tuning)

### Inference:
- **Real-time FPS**: 15-30 FPS (depends on hardware)
- **Model Sizes**: 
  - MobileNetV2: ~14 MB
  - EfficientNetB0: ~29 MB
  - ResNet50V2: ~98 MB

## 🔧 Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. GPU Support (Optional but Recommended)
For faster training, install CUDA-enabled TensorFlow:
```bash
pip install tensorflow-gpu>=2.10.0
```

## 💡 Best Practices

### Dataset Creation:
1. **Consistent Lighting**: Capture in well-lit environment
2. **Face Positioning**: Keep face centered and clearly visible
3. **Expression Variety**: Make genuine expressions for each emotion
4. **Multiple Sessions**: Capture at different times for variety
5. **Background**: Use neutral background when possible

### Training:
1. **GPU Recommended**: Training 3 models takes significant time on CPU
2. **Monitor Checkpoints**: Check `results/{model}/checkpoints/` for progress
3. **Review Logs**: Check log files if training seems stuck
4. **Resume if Needed**: Don't restart from scratch if interrupted
5. **Compare Models**: Use comparison report to select best model

### Monitoring:
1. **Good Lighting**: Improves detection accuracy
2. **Camera Distance**: Stay 1-2 feet from webcam
3. **Single Person**: System designed for one person at a time
4. **Regular Reviews**: Check CSV logs periodically
5. **Privacy**: Inform users about automated screenshot capture

## 🐛 Troubleshooting

### Dataset Issues:
**Issue**: "original dataset/train not found"
- **Solution**: Run `capture_dataset.py` first or ensure your dataset is in `original dataset/` folder

**Issue**: Unbalanced classes after augmentation
- **Solution**: Check if source images exist in all emotion folders

### Training Issues:
**Issue**: Out of memory during training
- **Solution**: Reduce `BATCH_SIZE` from 16 to 8 in training scripts

**Issue**: Training very slow
- **Solution**: 
  - Use GPU if available
  - Train one model at a time by commenting out others in `MODELS_CONFIG`
  - Reduce `EPOCHS` for testing

**Issue**: Checkpoint not loading
- **Solution**: Delete corrupted checkpoint and restart training

**Issue**: "No module named 'tensorflow'"
- **Solution**: `pip install tensorflow>=2.10.0`

### Webcam Issues:
**Issue**: Webcam not detected
- **Solution**: 
  - Check camera permissions
  - Ensure no other app is using webcam
  - Try different camera index: Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

**Issue**: Model file not found
- **Solution**: Ensure training completed successfully and `results/MobileNetV2/best_model.h5` exists

**Issue**: Low FPS during detection
- **Solution**: 
  - Use MobileNetV2 (fastest model)
  - Reduce camera resolution
  - Close other applications

**Issue**: Face not detected
- **Solution**: 
  - Improve lighting
  - Move closer to camera
  - Ensure face is clearly visible

## 🔍 System Logger (system_logger.py)

Comprehensive logging system used across all training and monitoring scripts.

**Features:**
- **Dual Output**: Logs to both file and console simultaneously
- **Timestamped Files**: Each session creates a unique log file with timestamp
- **Structured Logging**: Organized sections, subsections, metrics, and progress tracking
- **Exception Handling**: Captures full tracebacks for debugging
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

**Usage in Scripts:**
```python
from system_logger import SystemLogger

# Create logger
logger = SystemLogger(log_name="training", log_dir="logs")

# Log different types of information
logger.info("Starting training process")
logger.section("MODEL TRAINING")
logger.metric("Validation Accuracy", "92.5%")
logger.progress(10, 40, "Training epoch 10/40")
logger.error("Failed to load checkpoint", exc_info=True)

# Close logger
logger.close()
```

**Log File Format:**
```
2024-01-15 14:30:22 | INFO     | train_model          | Starting training process
2024-01-15 14:30:23 | DEBUG    | load_data            | Loading dataset from balanced_train/
2024-01-15 14:30:25 | INFO     | train_model          | Epoch 1/40 - Loss: 0.4523, Acc: 0.8234
2024-01-15 14:30:26 | WARNING  | save_checkpoint      | Checkpoint directory not found, creating...
2024-01-15 14:30:27 | ERROR    | load_model           | Model file not found
```

**Log Locations:**
- Training logs: `results/{model_name}/logs/training_*.log`
- System logs: `logs/system_*.log`
- All logs include timestamp in filename for easy tracking

**Benefits:**
- **Debugging**: Full traceback of errors for troubleshooting
- **Monitoring**: Track training progress without watching console
- **Audit Trail**: Complete record of all operations
- **Performance Analysis**: Review metrics and timing information

## 📝 Additional Notes

### Technical Details:
- **Transfer Learning**: All models use ImageNet pre-trained weights for better generalization
- **Data Augmentation**: Prevents overfitting on small datasets
- **Prediction Smoothing**: Reduces jitter in real-time detection (10-frame average)
- **Emotion Mapping**: 7 emotions → 2 stress categories for practical application
- **Memory Efficient**: Uses generators to avoid loading entire dataset into RAM
- **Comprehensive Logging**: SystemLogger tracks all operations with timestamps and structured output

### Privacy & Ethics:
- **User Consent**: Always inform users before monitoring
- **Data Storage**: Screenshots and logs stored locally
- **Transparency**: CSV logs provide full audit trail
- **Purpose**: Designed for stress awareness, not surveillance

### Customization:
- **Emotions**: Modify `EMOTIONS` list in scripts to add/remove classes
- **Stress Mapping**: Edit `STRESS_MAPPING` dict to change emotion-to-stress mapping
- **Screenshot Interval**: Change `SCREENSHOT_INTERVAL` in webcam script (default: 60s)
- **Batch Size**: Adjust `BATCH_SIZE` based on available memory
- **Models**: Add/remove models in training scripts
- **Logging**: Customize log levels and formats in system_logger.py

### File Locations:
- **Models**: `results/{model_name}/best_model.h5`
- **Training Logs**: `results/{model_name}/logs/training_*.log`
- **System Logs**: `logs/system_*.log`
- **Checkpoints**: `results/{model_name}/checkpoints/`
- **User Data**: `stress_monitoring/{username}/`
- **Reports**: `results/FINAL_MODEL_COMPARISON_REPORT.pdf`

## 🤝 Contributing

To improve the system:
1. Capture more diverse datasets
2. Experiment with different model architectures
3. Adjust hyperparameters for better performance
4. Add new emotion classes
5. Implement additional features

## 📄 License

This project is for educational and research purposes.
