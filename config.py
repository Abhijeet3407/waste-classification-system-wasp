"""
Configuration file for waste classification system
Contains all hyperparameters and settings
"""

import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.absolute()

# ==================== PATHS ====================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FINAL_MODELS_DIR = MODELS_DIR / "final"

LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, AUGMENTED_DATA_DIR,
                  MODELS_DIR, CHECKPOINTS_DIR, FINAL_MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== DATASET ====================
DATASET_NAME = "TrashNet"  # Options: TrashNet, WasteClassification
DATASET_URL = "https://github.com/garythung/trashnet/raw/master/data/"

# Class names (modify based on your dataset)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'organic', 'e-waste', 'hazardous']
NUM_CLASSES = len(CLASS_NAMES)

# Alternative simplified classes
SIMPLIFIED_CLASSES = ['plastic', 'paper', 'glass', 'metal', 'organic']
USE_SIMPLIFIED = False  # Set to True to use simplified classes

# ==================== IMAGE SETTINGS ====================
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ==================== DATA SPLIT ====================
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ==================== DATA AUGMENTATION ====================
APPLY_AUGMENTATION = True
AUGMENTATION_PARAMS = {
    'rotation_range': 30,        # wider rotation — helps custom CNN see more variance
    'width_shift_range': 0.25,
    'height_shift_range': 0.25,
    'horizontal_flip': True,
    'vertical_flip': False,
    'zoom_range': 0.2,           # slightly wider zoom
    'brightness_range': [0.7, 1.3],
    'fill_mode': 'nearest',
    'shear_range': 15            # stronger shear for extra diversity
}

# ==================== MODEL SETTINGS ====================
# Available models: 'custom_cnn', 'resnet50', 'mobilenetv2', 'efficientnetb0'
MODEL_TYPE = 'mobilenetv2'

# Transfer Learning
USE_TRANSFER_LEARNING = True
FREEZE_BASE_MODEL = False  # Fine-tune all layers
FINE_TUNE_LAYERS = 50

# ==================== TRAINING HYPERPARAMETERS ====================
BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LEARNING_RATE = 0.0001  # Lower LR for fine-tuning pretrained weights
OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'rmsprop'

# Learning Rate Schedule
USE_LR_SCHEDULE = True
LR_SCHEDULE_TYPE = 'reduce_on_plateau'  # Options: 'reduce_on_plateau', 'exponential', 'cosine'
LR_PATIENCE = 5
LR_FACTOR = 0.5
LR_MIN = 1e-7

# Early Stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.001

# Model Checkpoint
SAVE_BEST_ONLY = True
CHECKPOINT_MONITOR = 'val_accuracy'
CHECKPOINT_MODE = 'max'

# ==================== EVALUATION ====================
CONFIDENCE_THRESHOLD = 0.5
TOP_K_PREDICTIONS = 3

# ==================== EXPLAINABILITY ====================
USE_GRADCAM = True
GRADCAM_LAYER_NAME = 'auto'  # Will auto-detect last conv layer

# ==================== HARDWARE ====================
USE_GPU = True
MIXED_PRECISION = False  # Set to True for faster training on compatible GPUs
NUM_WORKERS = 4
PREFETCH_SIZE = 2

# ==================== LOGGING ====================
VERBOSE = 1
LOG_FREQUENCY = 10  # Log every N batches
TENSORBOARD_ENABLED = True
WANDB_ENABLED = False  # Set to True if using Weights & Biases

# ==================== API/WEB SETTINGS ====================
API_HOST = '0.0.0.0'
API_PORT = 8090
UPLOAD_FOLDER = PROJECT_ROOT / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16 MB

# ==================== REPRODUCIBILITY ====================
def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# ==================== DISPLAY SETTINGS ====================
def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("WASTE CLASSIFICATION SYSTEM - CONFIGURATION")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Transfer Learning: {USE_TRANSFER_LEARNING}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {INITIAL_LEARNING_RATE}")
    print(f"Augmentation: {APPLY_AUGMENTATION}")
    print(f"GPU Enabled: {USE_GPU}")
    print("=" * 60)
