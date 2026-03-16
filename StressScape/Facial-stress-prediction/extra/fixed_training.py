# Fixed training configuration for stress prediction model

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Load dataset
df = pd.read_csv("facial_expression.csv")

# Convert pixel string → numpy array
X = np.array([np.fromstring(pixels, sep=' ') for pixels in df['pixels']])
X = X.reshape(-1, 48, 48, 1)
X = np.repeat(X, 3, axis=-1)  # Convert to 3 channels
X = X / 255.0  # Normalize

# Labels
y = df['emotion'].values
y = to_categorical(y)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['emotion']
)

# Class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['emotion']),
    y=df['emotion']
)
class_weight_dict = dict(enumerate(class_weights))

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)

# Build model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(48,48,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# FIXED: Higher learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # Increased from 1e-5
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# FIXED: Simplified callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    mode='max'
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    mode='min'
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# FIXED: Larger batch size and more steps
batch_size = 16
steps_per_epoch = len(X_train) // batch_size

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weight_dict
)

print("Training completed!")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")