"""
Test Script - Validate System Setup
Quick tests to ensure everything is working correctly
"""

import sys
import os

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_imports():
    """Test if all required packages are installed"""
    print_section("Testing Package Imports")
    
    packages = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'flask': 'Flask',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    all_good = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20s} - OK")
        except ImportError:
            print(f"✗ {name:20s} - MISSING")
            all_good = False
    
    return all_good

def test_tensorflow_gpu():
    """Test if TensorFlow can access GPU"""
    print_section("Testing TensorFlow GPU Support")
    
    try:
        import tensorflow as tf
        
        print(f"TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU(s) detected: {len(gpus)}")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print("⚠ No GPU detected - will use CPU (slower but works)")
        
        return True
    except Exception as e:
        print(f"✗ Error testing TensorFlow: {e}")
        return False

def test_project_structure():
    """Test if project directories exist"""
    print_section("Testing Project Structure")
    
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'models/checkpoints',
        'models/final',
        'utils',
        'train',
        'evaluate',
        'predict',
        'frontend',
        'logs',
        'results',
        'uploads'
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path:30s} - EXISTS")
        else:
            print(f"✗ {dir_path:30s} - MISSING")
            all_good = False
    
    return all_good

def test_config():
    """Test if config can be imported"""
    print_section("Testing Configuration")
    
    try:
        import config
        print(f"✓ Config imported successfully")
        print(f"  Model type: {config.MODEL_TYPE}")
        print(f"  Classes: {len(config.CLASS_NAMES)}")
        print(f"  Epochs: {config.EPOCHS}")
        print(f"  Batch size: {config.BATCH_SIZE}")
        return True
    except Exception as e:
        print(f"✗ Error importing config: {e}")
        return False

def test_data_loader():
    """Test if data loader can be imported"""
    print_section("Testing Data Loader")
    
    try:
        from utils.data_loader import DataLoader
        loader = DataLoader()
        print(f"✓ DataLoader initialized")
        print(f"  Image size: {loader.img_size}")
        print(f"  Classes: {loader.class_names}")
        return True
    except Exception as e:
        print(f"✗ Error with DataLoader: {e}")
        return False

def test_model_builder():
    """Test if model builder works"""
    print_section("Testing Model Builder")
    
    try:
        from utils.model_builder import ModelBuilder
        builder = ModelBuilder(model_type='custom_cnn')
        print(f"✓ ModelBuilder initialized")
        print(f"  Model type: {builder.model_type}")
        print(f"  Input shape: {builder.input_shape}")
        print(f"  Num classes: {builder.num_classes}")
        return True
    except Exception as e:
        print(f"✗ Error with ModelBuilder: {e}")
        return False

def test_dataset():
    """Check if dataset exists"""
    print_section("Testing Dataset")
    
    from pathlib import Path
    import config
    
    raw_data = Path(config.RAW_DATA_DIR)
    
    if not raw_data.exists():
        print("✗ Raw data directory not found")
        print(f"  Expected: {raw_data}")
        print("\n  Run: python download_dataset.py")
        return False
    
    class_folders = [d for d in raw_data.iterdir() if d.is_dir()]
    
    if len(class_folders) == 0:
        print("✗ No class folders found")
        print("\n  Run: python download_dataset.py")
        return False
    
    print(f"✓ Dataset found with {len(class_folders)} classes:")
    total_images = 0
    for folder in class_folders:
        num_images = len(list(folder.glob("*.*")))
        total_images += num_images
        print(f"  - {folder.name:15s}: {num_images:4d} images")
    
    print(f"\nTotal images: {total_images}")
    
    # Check processed data
    processed_data = Path(config.PROCESSED_DATA_DIR)
    if (processed_data / 'X_train.npy').exists():
        print("✓ Processed data exists")
    else:
        print("⚠ Processed data not found")
        print("  Run: python utils/data_loader.py")
    
    return True

def test_model():
    """Check if trained model exists"""
    print_section("Testing Trained Model")
    
    from pathlib import Path
    import config
    
    model_files = list(config.FINAL_MODELS_DIR.glob("*.keras"))
    
    if not model_files:
        model_files = list(config.CHECKPOINTS_DIR.glob("*.keras"))
    
    if model_files:
        print(f"✓ Found {len(model_files)} trained model(s):")
        for model_file in model_files[:3]:  # Show max 3
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  - {model_file.name} ({size_mb:.1f} MB)")
        return True
    else:
        print("⚠ No trained models found")
        print("  Run: python train/train.py")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "█"*60)
    print("  WASTE CLASSIFICATION SYSTEM - SETUP VALIDATION")
    print("█"*60)
    
    results = {
        'Package Imports': test_imports(),
        'TensorFlow GPU': test_tensorflow_gpu(),
        'Project Structure': test_project_structure(),
        'Configuration': test_config(),
        'Data Loader': test_data_loader(),
        'Model Builder': test_model_builder(),
        'Dataset': test_dataset(),
        'Trained Model': test_model()
    }
    
    # Summary
    print_section("Test Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} - {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    # Recommendations
    print_section("Recommendations")
    
    if results['Package Imports']:
        print("✓ All packages installed correctly")
    else:
        print("⚠ Install missing packages:")
        print("  pip install -r requirements.txt")
    
    if not results['Dataset']:
        print("\n⚠ Dataset not ready:")
        print("  python download_dataset.py")
    
    if not results['Trained Model']:
        print("\n⚠ No trained model found:")
        print("  python train/train.py")
    
    if all(results.values()):
        print("\n" + "🎉"*30)
        print("  ALL TESTS PASSED! System is ready to use!")
        print("🎉"*30)
        print("\nNext steps:")
        print("  1. Train a model: python train/train.py")
        print("  2. Evaluate: python evaluate/evaluate.py")
        print("  3. Start web app: python app.py")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    run_all_tests()
