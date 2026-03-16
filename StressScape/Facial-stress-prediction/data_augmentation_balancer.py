import os
import shutil
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from collections import defaultdict

# Configuration
BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / "original dataset" / "train"
VAL_DIR = BASE_DIR / "original dataset" / "validation"
OUTPUT_TRAIN_DIR = BASE_DIR / "balanced_train"
OUTPUT_VAL_DIR = BASE_DIR / "balanced_validation"

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
TARGET_SIZE = (224, 224)

def count_images(directory):
    """Count images in each emotion folder"""
    counts = {}
    for emotion in EMOTIONS:
        emotion_path = directory / emotion
        if emotion_path.exists():
            counts[emotion] = len(list(emotion_path.glob('*.jpg'))) + len(list(emotion_path.glob('*.png')))
    return counts

def augment_images(source_dir, target_dir, emotion, num_to_generate):
    """Generate augmented images for a specific emotion"""
    source_path = source_dir / emotion
    target_path = target_dir / emotion
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Copy original images first
    for img_file in source_path.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            shutil.copy2(img_file, target_path / img_file.name)
    
    if num_to_generate <= 0:
        return
    
    # Data augmentation configuration
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Get all original images
    image_files = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
    
    generated = 0
    while generated < num_to_generate:
        for img_file in image_files:
            if generated >= num_to_generate:
                break
            
            img = load_img(img_file, target_size=TARGET_SIZE)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            i = 0
            for batch in datagen.flow(x, batch_size=1):
                aug_img_name = f"{img_file.stem}_aug_{generated}_{i}.jpg"
                aug_img_path = target_path / aug_img_name
                
                # Save augmented image
                from tensorflow.keras.utils import array_to_img
                aug_img = array_to_img(batch[0])
                aug_img.save(aug_img_path)
                
                generated += 1
                i += 1
                if generated >= num_to_generate or i >= 3:
                    break
    
    print(f"  Generated {generated} augmented images for {emotion}")

def balance_dataset():
    """Balance the dataset by augmenting minority classes"""
    print("=" * 60)
    print("DATASET BALANCING AND AUGMENTATION")
    print("=" * 60)
    
    # Count images in train and validation
    print("\n1. Counting images in original dataset...")
    train_counts = count_images(TRAIN_DIR)
    val_counts = count_images(VAL_DIR)
    
    print("\nOriginal Training Set:")
    for emotion, count in sorted(train_counts.items()):
        print(f"  {emotion:12s}: {count:5d} images")
    
    print("\nOriginal Validation Set:")
    for emotion, count in sorted(val_counts.items()):
        print(f"  {emotion:12s}: {count:5d} images")
    
    # Find maximum class
    max_train = max(train_counts.values())
    max_val = max(val_counts.values())
    
    print(f"\n2. Target counts:")
    print(f"  Training: {max_train} images per class")
    print(f"  Validation: {max_val} images per class")
    
    # Create output directories
    OUTPUT_TRAIN_DIR.mkdir(exist_ok=True)
    OUTPUT_VAL_DIR.mkdir(exist_ok=True)
    
    # Balance training set
    print("\n3. Balancing training set...")
    for emotion in EMOTIONS:
        current_count = train_counts[emotion]
        needed = max_train - current_count
        print(f"\n  Processing {emotion}:")
        print(f"    Current: {current_count}, Target: {max_train}, Need: {needed}")
        augment_images(TRAIN_DIR, OUTPUT_TRAIN_DIR, emotion, needed)
    
    # Balance validation set
    print("\n4. Balancing validation set...")
    for emotion in EMOTIONS:
        current_count = val_counts[emotion]
        needed = max_val - current_count
        print(f"\n  Processing {emotion}:")
        print(f"    Current: {current_count}, Target: {max_val}, Need: {needed}")
        augment_images(VAL_DIR, OUTPUT_VAL_DIR, emotion, needed)
    
    # Final count
    print("\n5. Final counts:")
    final_train_counts = count_images(OUTPUT_TRAIN_DIR)
    final_val_counts = count_images(OUTPUT_VAL_DIR)
    
    print("\nBalanced Training Set:")
    for emotion, count in sorted(final_train_counts.items()):
        print(f"  {emotion:12s}: {count:5d} images")
    
    print("\nBalanced Validation Set:")
    for emotion, count in sorted(final_val_counts.items()):
        print(f"  {emotion:12s}: {count:5d} images")
    
    print("\n" + "=" * 60)
    print("DATASET BALANCING COMPLETED!")
    print("=" * 60)
    print(f"\nBalanced datasets saved to:")
    print(f"  Training: {OUTPUT_TRAIN_DIR}")
    print(f"  Validation: {OUTPUT_VAL_DIR}")

if __name__ == "__main__":
    balance_dataset()
