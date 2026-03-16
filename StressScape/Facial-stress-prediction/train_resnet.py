import numpy as np
import tensorflow as tf
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50V2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc as sklearn_auc
import json
from system_logger import SystemLogger
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime

def build_model():
    """Build ResNet50V2 model"""
    base_model = ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
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
    
    model = Model(inputs=base_model.input, outputs=outputs, name='ResNet50V2_Stress')
    return model, base_model

def generate_individual_report(model_name, history, y_true, y_pred, y_pred_probs, metrics):
    """Generate individual PDF report with plots"""
    model_dir = RESULTS_DIR
    
    # Generate confusion matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Not Stressed', 'Stressed'],
                yticklabels=['Not Stressed', 'Stressed'])
    axes[0].set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs.flatten())
    roc_auc = sklearn_auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(model_dir / 'confusion_matrix_roc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Training history plots (only if we have history)
    if history.history.get('accuracy') and len(history.history['accuracy']) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Accuracy Over Epochs', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Loss Over Epochs', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC - handle both metric name formats
        auc_key = 'auc_1' if 'auc_1' in history.history else 'auc'
        val_auc_key = 'val_auc_1' if 'val_auc_1' in history.history else 'val_auc'
        
        if auc_key in history.history and val_auc_key in history.history:
            axes[1, 0].plot(history.history[auc_key], label='Train', linewidth=2)
            axes[1, 0].plot(history.history[val_auc_key], label='Validation', linewidth=2)
            axes[1, 0].set_title('AUC Over Epochs', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'AUC data not available', ha='center', va='center')
        
        # Precision & Recall
        prec_key = 'precision_1' if 'precision_1' in history.history else 'precision'
        val_prec_key = 'val_precision_1' if 'val_precision_1' in history.history else 'val_precision'
        recall_key = 'recall_1' if 'recall_1' in history.history else 'recall'
        val_recall_key = 'val_recall_1' if 'val_recall_1' in history.history else 'val_recall'
        
        if all(k in history.history for k in [prec_key, val_prec_key, recall_key, val_recall_key]):
            axes[1, 1].plot(history.history[prec_key], label='Train Precision', linewidth=2)
            axes[1, 1].plot(history.history[val_prec_key], label='Val Precision', linewidth=2)
            axes[1, 1].plot(history.history[recall_key], label='Train Recall', linewidth=2)
            axes[1, 1].plot(history.history[val_recall_key], label='Val Recall', linewidth=2)
            axes[1, 1].set_title('Precision & Recall Over Epochs', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Precision/Recall data not available', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(model_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate PDF
    pdf_path = model_dir / f'{model_name}_training_report.pdf'
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph(f"{model_name} Training Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    summary = f"""
    <b>Training Configuration:</b><br/>
    • Model: {model_name}<br/>
    • Training Epochs: 30 epochs<br/>
    • Fine-tuning Epochs: 10 epochs<br/>
    • Total Epochs: 40 epochs<br/>
    • Image Size: 224x224<br/>
    • Batch Size: {BATCH_SIZE}<br/>
    • Optimizer: Adam<br/>
    • Loss Function: Binary Crossentropy<br/><br/>
    
    <b>Final Performance Metrics:</b><br/>
    • Best Validation Accuracy: {metrics['best_val_accuracy']:.4f}<br/>
    • Final Validation Accuracy: {metrics['final_val_accuracy']:.4f}<br/>
    • Final Validation Loss: {metrics['final_val_loss']:.4f}<br/>
    • Precision: {metrics['calculated_precision']:.4f}<br/>
    • Recall: {metrics['calculated_recall']:.4f}<br/>
    • F1-Score: {metrics['calculated_f1_score']:.4f}<br/>
    • AUC: {metrics['final_val_auc']:.4f}<br/><br/>
    
    <b>Dataset Distribution:</b><br/>
    • Stressed Samples: {metrics['stressed_samples']}<br/>
    • Not Stressed Samples: {metrics['not_stressed_samples']}<br/>
    • Total Samples: {metrics['stressed_samples'] + metrics['not_stressed_samples']}<br/>
    """
    story.append(Paragraph(summary, styles['Normal']))
    story.append(PageBreak())
    
    story.append(Paragraph("Training History", styles['Heading1']))
    story.append(Spacer(1, 12))
    if (model_dir / 'training_history.png').exists():
        story.append(RLImage(str(model_dir / 'training_history.png'), width=6.5*inch, height=5.4*inch))
    story.append(PageBreak())
    
    story.append(Paragraph("Evaluation Results", styles['Heading1']))
    story.append(Spacer(1, 12))
    if (model_dir / 'confusion_matrix_roc.png').exists():
        story.append(RLImage(str(model_dir / 'confusion_matrix_roc.png'), width=6.5*inch, height=2.7*inch))
    story.append(Spacer(1, 20))
    
    report_file = model_dir / 'classification_report.txt'
    if report_file.exists():
        with open(report_file, 'r') as f:
            report_text = f.read()
        story.append(Paragraph("Classification Report", styles['Heading2']))
        story.append(Spacer(1, 12))
        for line in report_text.split('\n')[:15]:
            if line.strip():
                story.append(Paragraph(f"<font face='Courier' size=8>{line}</font>", styles['Code']))
    
    doc.build(story)
    print(f"  ✓ Individual PDF report: {pdf_path}")

# Configuration
BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / "balanced_train"
VAL_DIR = BASE_DIR / "balanced_validation"
RESULTS_DIR = BASE_DIR / "results" / "ResNet50V2"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 30
FINE_TUNE_EPOCHS = 10

STRESS_MAPPING = {'angry': 1, 'sad': 1, 'fear': 1, 'disgust': 1, 'neutral': 1, 'happy': 0, 'surprise': 0}

class StressDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_generator, stress_mapping):
        self.image_generator = image_generator
        self.emotion_to_stress = {idx: stress_mapping.get(emotion, 0) 
                                  for emotion, idx in image_generator.class_indices.items()}
        self.stress_labels = np.array([self.emotion_to_stress[idx] 
                                       for idx in image_generator.labels])
    
    def __len__(self):
        return len(self.image_generator)
    
    def __getitem__(self, idx):
        batch_x, batch_y = self.image_generator[idx]
        batch_y_stress = np.array([self.emotion_to_stress[np.argmax(label)] for label in batch_y])
        return batch_x, batch_y_stress
    
    def on_epoch_end(self):
        self.image_generator.on_epoch_end()
    
    def get_stress_labels(self):
        return self.stress_labels

def train_resnet():
    print("="*80)
    print("TRAINING ResNet50V2")
    print("="*80)
    
    # GPU memory management
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = SystemLogger('ResNet50V2_training', str(RESULTS_DIR / 'logs'))
    
    try:
        # Create generators
        train_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
            rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
            brightness_range=[0.8, 1.2], fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
        )
        
        train_flow = train_datagen.flow_from_directory(
            TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', shuffle=True, seed=42
        )
        val_flow = val_datagen.flow_from_directory(
            VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', shuffle=False
        )
        
        train_gen = StressDataGenerator(train_flow, STRESS_MAPPING)
        val_gen = StressDataGenerator(val_flow, STRESS_MAPPING)
        
        logger.info(f"Training samples: {len(train_gen.get_stress_labels())}")
        logger.info(f"Validation samples: {len(val_gen.get_stress_labels())}")
        
        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(train_gen.get_stress_labels()), 
                                            y=train_gen.get_stress_labels())
        class_weight_dict = {int(cls): float(weight) for cls, weight in enumerate(class_weights)}
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Check for checkpoints
        checkpoint_dir = RESULTS_DIR / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        phase2_weights = checkpoint_dir / 'best_weights_finetuned.h5'
        phase1_weights = checkpoint_dir / 'best_weights.h5'
        
        skip_phase1 = False
        model, base_model = build_model()
        
        # Try to load checkpoint
        if phase2_weights.exists() and phase2_weights.stat().st_size > 1000:
            response = input(f"\n✓ Found checkpoint: {phase2_weights.name}. Resume? (y/n): ").lower()
            if response == 'y':
                try:
                    model.load_weights(str(phase2_weights))
                    logger.info(f"Loaded checkpoint: {phase2_weights.name}")
                    skip_phase1 = True
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}")
        
        if not skip_phase1 and phase1_weights.exists() and phase1_weights.stat().st_size > 1000:
            response = input(f"\n✓ Found checkpoint: {phase1_weights.name}. Resume? (y/n): ").lower()
            if response == 'y':
                try:
                    model.load_weights(str(phase1_weights))
                    logger.info(f"Loaded checkpoint: {phase1_weights.name}")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}")
        
        model.compile(optimizer=Adam(learning_rate=0.0001 if not skip_phase1 else 1e-5), 
                     loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.Precision(), 
                             tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
        
        logger.info(f"Total parameters: {model.count_params():,}")
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint_dir / ('best_weights_finetuned.h5' if skip_phase1 else 'best_weights.h5')),
                monitor='val_accuracy', 
                save_best_only=True,
                save_weights_only=True,
                mode='max',
                verbose=1,
                save_freq='epoch'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True,
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.3, 
                patience=5, 
                min_lr=1e-7,
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.TerminateOnNaN()
        ]
        
        # Phase 1
        resume_epoch = 0
        if not skip_phase1:
            # Ask if resuming from specific epoch
            if phase1_weights.exists():
                resume_input = input("Resume from specific epoch? (Enter epoch number or 0 to start fresh): ")
                try:
                    resume_epoch = int(resume_input)
                    if resume_epoch > 0:
                        logger.info(f"Resuming Phase 1 from epoch {resume_epoch}")
                except ValueError:
                    resume_epoch = 0
            
            logger.info("Phase 1: Training classifier head")
            history1 = model.fit(train_gen, validation_data=val_gen, 
                               epochs=EPOCHS,
                               initial_epoch=resume_epoch,
                               callbacks=callbacks, class_weight=class_weight_dict, verbose=1)
        else:
            logger.info("Skipping Phase 1")
            history1 = type('obj', (object,), {'history': {
                'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 
                'auc_1': [], 'val_auc_1': [], 'precision_1': [], 'val_precision_1': [], 
                'recall_1': [], 'val_recall_1': []
            }})()
        
        # Phase 2: Fine-tuning
        logger.info("Phase 2: Fine-tuning")
        base_model.trainable = True
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        
        model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.Precision(), 
                             tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
        
        callbacks[0] = tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_dir / 'best_weights_finetuned.h5'),
            monitor='val_accuracy', save_best_only=True,
            save_weights_only=True, mode='max', verbose=1
        )
        
        history2 = model.fit(train_gen, validation_data=val_gen, 
                           epochs=EPOCHS + FINE_TUNE_EPOCHS,
                           initial_epoch=EPOCHS if not skip_phase1 else 0,
                           callbacks=callbacks, class_weight=class_weight_dict, verbose=1)
        
        # Combine histories
        for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
            if key in history1.history:
                history1.history[key].extend(history2.history[key])
        
        # Handle metrics with _1 suffix
        metric_pairs = [('auc_1', 'auc_1'), ('val_auc_1', 'val_auc_1'),
                       ('precision_1', 'precision_1'), ('val_precision_1', 'val_precision_1'),
                       ('recall_1', 'recall_1'), ('val_recall_1', 'val_recall_1')]
        
        for h1_key, h2_key in metric_pairs:
            if h1_key in history1.history and h2_key in history2.history:
                history1.history[h1_key].extend(history2.history[h2_key])
        
        # Evaluate
        logger.info("Final evaluation")
        y_pred_probs = model.predict(val_gen, verbose=1)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = val_gen.get_stress_labels()
        
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred, y_pred_probs = y_true[:min_len], y_pred[:min_len], y_pred_probs[:min_len]
        
        # Calculate metrics - handle both metric name formats
        val_auc_key = 'val_auc_1' if 'val_auc_1' in history1.history else 'val_auc'
        
        metrics = {
            'best_val_accuracy': float(max(history1.history['val_accuracy'])) if history1.history['val_accuracy'] else 0.0,
            'final_val_accuracy': float(history1.history['val_accuracy'][-1]) if history1.history['val_accuracy'] else 0.0,
            'final_val_loss': float(history1.history['val_loss'][-1]) if history1.history['val_loss'] else 0.0,
            'final_val_auc': float(history1.history[val_auc_key][-1]) if val_auc_key in history1.history and history1.history[val_auc_key] else 0.0,
            'calculated_accuracy': float(accuracy_score(y_true, y_pred)),
            'calculated_precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'calculated_recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'calculated_f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'total_epochs': len([x for x in history1.history['accuracy'] if x]) + len(history2.history['accuracy']),
            'stressed_samples': int(np.sum(y_true)),
            'not_stressed_samples': int(len(y_true) - np.sum(y_true))
        }
        
        with open(RESULTS_DIR / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info("Saving model...")
        
        # Save only weights (avoids serialization issues)
        try:
            model.save_weights(str(RESULTS_DIR / 'final_model_weights.h5'))
            logger.info(f"✓ Weights saved: {RESULTS_DIR / 'final_model_weights.h5'}")
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")
        
        # Try SavedModel format
        try:
            model.save(str(RESULTS_DIR / 'saved_model'), save_format='tf')
            logger.info(f"✓ SavedModel saved: {RESULTS_DIR / 'saved_model'}")
        except Exception as e:
            logger.warning(f"Failed to save in SavedModel format: {e}")
        
        # Save classification report
        report = classification_report(y_true, y_pred, target_names=['Not Stressed', 'Stressed'], zero_division=0)
        with open(RESULTS_DIR / 'classification_report.txt', 'w') as f:
            f.write(f"ResNet50V2 Classification Report\n{'='*60}\n\n{report}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix(y_true, y_pred)))
        
        # Generate reports
        logger.info("Generating reports...")
        generate_individual_report('ResNet50V2', history1, y_true, y_pred, y_pred_probs, metrics)
        
        print(f"\n{'='*80}")
        print("✓ ResNet50V2 training completed!")
        print(f"{'='*80}")
        print(f"  Accuracy:  {metrics['calculated_accuracy']:.4f}")
        print(f"  Precision: {metrics['calculated_precision']:.4f}")
        print(f"  Recall:    {metrics['calculated_recall']:.4f}")
        print(f"  F1-Score:  {metrics['calculated_f1_score']:.4f}")
        print(f"  AUC:       {metrics['final_val_auc']:.4f}")
        print(f"{'='*80}")
        print(f"  Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
    finally:
        logger.close()

if __name__ == "__main__":
    train_resnet()