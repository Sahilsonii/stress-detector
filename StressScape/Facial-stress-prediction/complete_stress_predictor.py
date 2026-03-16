# Complete Stress Prediction Model - Optimized for Small Dataset

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import os

# Load dataset
df = pd.read_csv("facial_expression.csv")
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['emotion'].value_counts()}")

# Convert pixel string → numpy array
X = np.array([np.fromstring(pixels, sep=' ') for pixels in df['pixels']])
X = X.reshape(-1, 48, 48, 1)
X = np.repeat(X, 3, axis=-1) / 255.0
y = df['emotion'].values
y_categorical = to_categorical(y)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y
)



# Focal loss to handle class imbalance
def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=1))
    return focal_loss_fixed

# Simpler model for small dataset
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(64, activation='relu'),
    Dropout(0.3),
    
    Dense(2, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss(alpha=0.75, gamma=2.0),  # Use focal loss for better class balance
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=16,
    epochs=50,
    class_weight={0: 1.0, 1: 3.0},
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint('best_stress_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
)

# Evaluation with adjusted threshold
y_pred = model.predict(X_val)

# Adjust decision threshold to balance predictions
threshold = 0.4  # Lower threshold favors stress class
y_pred_classes = (y_pred[:, 1] > threshold).astype(int)
y_true_classes = np.argmax(y_val, axis=1)

print(f"\nUsing decision threshold: {threshold}")

# Generate plots
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')

# Confusion matrix
plt.subplot(1, 3, 3)
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Stress', 'Stress'], 
            yticklabels=['No Stress', 'Stress'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Print results
final_val_acc = max(history.history['val_accuracy'])
print(f"\nFinal Results:")
print(f"Best Validation Accuracy: {final_val_acc:.4f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

# Check prediction distribution
unique_pred, counts_pred = np.unique(y_pred_classes, return_counts=True)
print(f"Prediction distribution: {dict(zip(unique_pred, counts_pred))}")
unique_true, counts_true = np.unique(y_true_classes, return_counts=True)
print(f"True distribution: {dict(zip(unique_true, counts_true))}")

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, 
                          target_names=['No Stress', 'Stress'], zero_division=0))

# Generate PDF Report
def create_pdf_report():
    doc = SimpleDocTemplate("stress_prediction_report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Stress Prediction Model Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Model Summary
    pred_dist = dict(zip(*np.unique(y_pred_classes, return_counts=True)))
    summary_text = f"""
    <b>Model Performance Summary:</b><br/>
    • Best Validation Accuracy: {final_val_acc:.4f}<br/>
    • Final Training Accuracy: {history.history['accuracy'][-1]:.4f}<br/>
    • Total Training Epochs: {len(history.history['accuracy'])}<br/>
    • Dataset Size: {len(df)} samples<br/>
    • Training Samples: {len(X_train)}<br/>
    • Validation Samples: {len(X_val)}<br/>
    • Decision Threshold: 0.4<br/>
    • Predictions: No Stress={pred_dist.get(0, 0)}, Stress={pred_dist.get(1, 0)}<br/>
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add plots
    if os.path.exists('training_results.png'):
        img = RLImage('training_results.png', width=500, height=167)
        story.append(img)
    
    # Classification report
    cr_text = classification_report(y_true_classes, y_pred_classes, 
                                  target_names=['No Stress', 'Stress'], zero_division=0)
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Classification Report:</b>", styles['Heading2']))
    story.append(Paragraph(f"<pre>{cr_text}</pre>", styles['Code']))
    
    doc.build(story)
    print("PDF report saved as 'stress_prediction_report.pdf'")

create_pdf_report()
print("Training completed with optimized model and report generated!")