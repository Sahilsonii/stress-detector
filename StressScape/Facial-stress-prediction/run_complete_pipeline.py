import subprocess
import sys
import os
from pathlib import Path

def main():
    print("="*70)
    print("COMPLETE STRESS DETECTION PIPELINE")
    print("="*70)
    print("\nThis script will run the complete pipeline:")
    print("0. Capture dataset (optional)")
    print("1. Balance dataset with augmentation")
    print("2. Train models (individual scripts)")
    print("   2a. Train MobileNetV2")
    print("   2b. Train EfficientNetB0")
    print("   2c. Train ResNet50V2")
    print("3. Generate comparison report")
    print("4. Regenerate reports only (no training)")
    print("5. Launch webcam stress detector")
    print("\n" + "="*70)
    
    base_dir = Path(__file__).parent
    
    def run_command(command, description):
        """Run a command and handle errors"""
        print("\n" + "="*70)
        print(f"STEP: {description}")
        print("="*70)
        
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, cwd=str(base_dir))
            print(f"✓ {description} completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error during {description}")
            print(f"Error: {e}")
            return False
    
    # Step 0: Capture dataset
    response = input("\nRun Step 0: Capture dataset? (y/n): ").lower()
    if response == 'y':
        if not run_command(f"python \"{base_dir / 'capture_dataset.py'}\"", "Dataset Capture"):
            print("\nPipeline stopped due to error.")
            return
    else:
        print("Skipping Step 0...")
    
    # Check if train and validation folders exist
    train_dir = base_dir / "original dataset" / "train"
    val_dir = base_dir / "original dataset" / "validation"
    
    if not train_dir.exists() or not val_dir.exists():
        print("\n✗ Error: 'original dataset/train' and 'original dataset/validation' folders not found!")
        print("Please ensure your dataset is in the correct location or run Step 0 to capture dataset.")
        return
    
    # Step 1: Balance dataset
    response = input("\nRun Step 1: Balance dataset? (y/n): ").lower()
    if response == 'y':
        script_path = base_dir / "data_augmentation_balancer.py"
        if not run_command(f"python \"{script_path}\"", "Dataset Balancing"):
            print("\nPipeline stopped due to error.")
            return
    else:
        print("Skipping Step 1...")
    
    # Step 2: Train models
    balanced_train = base_dir / "balanced_train"
    balanced_val = base_dir / "balanced_validation"
    
    if not balanced_train.exists() or not balanced_val.exists():
        print("\n✗ Error: Balanced datasets not found!")
        print("Please run Step 1 first.")
        return
    
    # Train MobileNetV2
    response = input("\nRun Step 2a: Train MobileNetV2? (y/n): ").lower()
    if response == 'y':
        if not run_command(f"python \"{base_dir / 'train_mobilenet.py'}\"", "MobileNetV2 Training"):
            print("\nContinuing to next model...")
    else:
        print("Skipping MobileNetV2...")
    
    # Train EfficientNetB0
    response = input("\nRun Step 2b: Train EfficientNetB0? (y/n): ").lower()
    if response == 'y':
        if not run_command(f"python \"{base_dir / 'train_efficientnet.py'}\"", "EfficientNetB0 Training"):
            print("\nContinuing to next model...")
    else:
        print("Skipping EfficientNetB0...")
    
    # Train ResNet50V2
    response = input("\nRun Step 2c: Train ResNet50V2? (y/n): ").lower()
    if response == 'y':
        if not run_command(f"python \"{base_dir / 'train_resnet.py'}\"", "ResNet50V2 Training"):
            print("\nContinuing...")
    else:
        print("Skipping ResNet50V2...")
    
    # Step 3: Generate comparison report
    response = input("\nRun Step 3: Generate comparison report? (y/n): ").lower()
    if response == 'y':
        run_command(f"python \"{base_dir / 'generate_comparison_report.py'}\"", "Comparison Report Generation")
    else:
        print("Skipping Step 3...")
    
    # Step 4: Regenerate reports only
    response = input("\nRun Step 4: Regenerate reports only (no training)? (y/n): ").lower()
    if response == 'y':
        run_command(f"python \"{base_dir / 'regenerate_reports_only.py'}\"", "Report Regeneration")
    else:
        print("Skipping Step 4...")
    
    # Step 5: Webcam detection
    response = input("\nRun Step 5: Launch webcam detector? (y/n): ").lower()
    if response == 'y':
        basic_model = base_dir / "best_stress_detector.h5"
        comprehensive_models = base_dir / "results"
        
        if not basic_model.exists() and not comprehensive_models.exists():
            print("\n✗ Error: No trained models found!")
            print("Please run Step 2 to train models first.")
            return
        
        print("\nLaunching webcam stress detector...")
        print("Press 'q' in the webcam window to quit")
        script_path = base_dir / "webcam_stress_detector.py"
        run_command(f"python \"{script_path}\"", "Webcam Stress Detection")
    else:
        print("Skipping Step 5...")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED!")
    print("="*70)
    print(f"\nGenerated files in: {base_dir}")
    print("  • dataset/ - Captured dataset (if Step 0 was run)")
    print("  • balanced_train/ - Augmented training dataset")
    print("  • balanced_validation/ - Augmented validation dataset")
    print("  • results/MobileNetV2/ - MobileNetV2 model and results")
    print("  • results/EfficientNetB0/ - EfficientNetB0 model and results")
    print("  • results/ResNet50V2/ - ResNet50V2 model and results")
    print("  • results/comparison/ - Comparison plots")
    print("  • results/FINAL_MODEL_COMPARISON_REPORT.pdf - Full report")
    print("  • stress_monitoring/ - Webcam monitoring logs")

if __name__ == "__main__":
    main()
