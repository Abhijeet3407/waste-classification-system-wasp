#!/usr/bin/env python3
"""
Quick Start Script for Waste Classification System
Automates the entire setup and training process
"""

import os
import sys
from pathlib import Path
import subprocess
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"▶ {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {description} failed")
        print(f"Error message: {e.stderr}")
        return False

def check_requirements():
    """Check if requirements are installed"""
    print_header("Checking Requirements")
    
    try:
        import tensorflow
        import numpy
        import pandas
        import sklearn
        import PIL
        import flask
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e.name}")
        print("\nPlease install requirements first:")
        print("pip install -r requirements.txt")
        return False

def check_dataset():
    """Check if dataset exists"""
    print_header("Checking Dataset")
    
    raw_data_dir = Path("data/raw")
    
    if not raw_data_dir.exists():
        print("✗ Dataset directory not found")
        print("\nPlease organize your dataset in data/raw/ with class folders:")
        print("data/raw/")
        print("  ├── class1/")
        print("  ├── class2/")
        print("  └── ...")
        return False
    
    # Check for class folders
    class_folders = [d for d in raw_data_dir.iterdir() if d.is_dir()]
    
    if len(class_folders) == 0:
        print("✗ No class folders found in data/raw/")
        return False
    
    print(f"✓ Found {len(class_folders)} class folders:")
    for folder in class_folders:
        num_images = len(list(folder.glob("*.*")))
        print(f"  - {folder.name}: {num_images} images")
    
    return True

def prepare_data():
    """Run data preparation"""
    print_header("Preparing Data")
    
    if Path("data/processed/X_train.npy").exists():
        response = input("Processed data already exists. Reprocess? (y/N): ")
        if response.lower() != 'y':
            print("Skipping data preparation")
            return True
    
    return run_command(
        "python utils/data_loader.py",
        "Data preprocessing"
    )

def train_model():
    """Train the model"""
    print_header("Training Model")
    
    response = input("Start training? This may take 10-60 minutes. (Y/n): ")
    if response.lower() == 'n':
        print("Skipping model training")
        return True
    
    return run_command(
        "python train/train.py",
        "Model training"
    )

def evaluate_model():
    """Evaluate the model"""
    print_header("Evaluating Model")
    
    return run_command(
        "python evaluate/evaluate.py",
        "Model evaluation"
    )

def test_prediction():
    """Test prediction on a sample image"""
    print_header("Testing Prediction")
    
    # Find a test image
    test_images = list(Path("data/raw").rglob("*.jpg"))[:1]
    
    if not test_images:
        print("No test images found, skipping prediction test")
        return True
    
    test_image = test_images[0]
    print(f"Testing with: {test_image}")
    
    return run_command(
        f"python predict/predict.py {test_image}",
        "Prediction test"
    )

def show_results():
    """Show results summary"""
    print_header("Setup Complete!")
    
    print("Your waste classification system is ready!\n")
    print("Next steps:\n")
    print("1. View results in the 'results/' directory")
    print("2. Check training plots and metrics")
    print("3. Start the web application:")
    print("   python app.py")
    print("4. Access the web interface at http://localhost:8080\n")
    print("For predictions:")
    print("   python predict/predict.py <image_path>")
    print("\nFor explainability:")
    print("   python predict/explainability.py <image_path>\n")
    
    response = input("Start web application now? (y/N): ")
    if response.lower() == 'y':
        print("\nStarting web application...")
        print("Access at: http://localhost:8080")
        print("Press Ctrl+C to stop\n")
        subprocess.run("python app.py", shell=True)

def main():
    """Main quick start function"""
    print_header("Waste Classification System - Quick Start")
    
    print("This script will help you set up and run the waste classification system.\n")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check dataset
    if not check_dataset():
        sys.exit(1)
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Full setup (prepare data + train + evaluate)")
    print("2. Just prepare data")
    print("3. Just train model (data must be prepared)")
    print("4. Just evaluate model (model must be trained)")
    print("5. Start web application")
    print("6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        # Full setup
        if not prepare_data():
            print("Data preparation failed. Exiting.")
            sys.exit(1)
        
        if not train_model():
            print("Training failed. Exiting.")
            sys.exit(1)
        
        if not evaluate_model():
            print("Evaluation failed. Exiting.")
            sys.exit(1)
        
        test_prediction()
        show_results()
    
    elif choice == '2':
        prepare_data()
    
    elif choice == '3':
        if not train_model():
            sys.exit(1)
        evaluate_model()
        show_results()
    
    elif choice == '4':
        evaluate_model()
        show_results()
    
    elif choice == '5':
        print("\nStarting web application...")
        print("Access at: http://localhost:8080")
        print("Press Ctrl+C to stop\n")
        subprocess.run("python app.py", shell=True)
    
    elif choice == '6':
        print("Goodbye!")
        sys.exit(0)
    
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)
