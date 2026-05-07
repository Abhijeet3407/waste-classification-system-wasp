"""
Data loading and preprocessing utilities
Handles dataset preparation, cleaning, and splitting
"""

import hashlib
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import imagehash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import shutil

# Allow running directly as a script from any working directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DataLoader:
    """Handle all data loading and preprocessing operations"""
    
    def __init__(self, data_path=None):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to raw data directory
        """
        self.data_path = Path(data_path) if data_path else config.RAW_DATA_DIR
        self.class_names = config.CLASS_NAMES
        self.img_size = config.IMG_SIZE
        self.label_encoder = LabelEncoder()
        
    def load_dataset_from_folders(self):
        """
        Load dataset organized in folders (one folder per class)
        Expected structure:
        data/raw/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
        
        Returns:
            images: numpy array of images
            labels: numpy array of labels
        """
        print("Loading dataset from folders...")
        
        images = []
        labels = []
        
        for class_name in tqdm(self.class_names, desc="Loading classes"):
            class_path = self.data_path / class_name
            
            if not class_path.exists():
                print(f"Warning: Class folder '{class_name}' not found at {class_path}")
                continue
            
            image_files = list(class_path.glob('*.*'))
            
            for img_file in tqdm(image_files, desc=f"Loading {class_name}", leave=False):
                try:
                    # Read and validate image
                    img = self._read_image(img_file)
                    if img is not None:
                        images.append(img)
                        labels.append(class_name)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue
        
        print(f"Loaded {len(images)} images from {len(set(labels))} classes")
        
        return np.array(images), np.array(labels)
    
    def load_dataset_from_csv(self, csv_path, image_col='filename', label_col='label'):
        """
        Load dataset from CSV file with image paths and labels
        
        Args:
            csv_path: Path to CSV file
            image_col: Column name for image paths
            label_col: Column name for labels
            
        Returns:
            images: numpy array of images
            labels: numpy array of labels
        """
        print(f"Loading dataset from CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
        images = []
        labels = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
            try:
                img_path = self.data_path / row[image_col]
                img = self._read_image(img_path)
                
                if img is not None:
                    images.append(img)
                    labels.append(row[label_col])
            except Exception as e:
                print(f"Error loading image at row {idx}: {e}")
                continue
        
        print(f"Loaded {len(images)} images")
        
        return np.array(images), np.array(labels)
    
    def _read_image(self, img_path):
        """
        Read and preprocess a single image
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed image or None if error
        """
        try:
            # Open image
            img = Image.open(img_path)
            
            # Convert to RGB (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target size
            img = img.resize(self.img_size, Image.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Validate image
            if img_array.shape != config.INPUT_SHAPE:
                print(f"Invalid shape {img_array.shape} for {img_path}")
                return None
            
            return img_array
            
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            return None
    
    def clean_corrupted_images(self, images, labels):
        """
        Remove corrupted or invalid images
        
        Args:
            images: numpy array of images
            labels: numpy array of labels
            
        Returns:
            Cleaned images and labels
        """
        print("Cleaning corrupted images...")
        
        valid_indices = []
        
        for idx, img in enumerate(tqdm(images, desc="Validating images")):
            # Check if image has valid shape
            if img.shape == config.INPUT_SHAPE:
                # Check if image has valid values
                if np.min(img) >= 0 and np.max(img) <= 255:
                    # Check if image is not all zeros
                    if np.sum(img) > 0:
                        valid_indices.append(idx)
        
        cleaned_images = images[valid_indices]
        cleaned_labels = labels[valid_indices]
        
        removed_count = len(images) - len(cleaned_images)
        print(f"Removed {removed_count} corrupted images")
        print(f"Valid images: {len(cleaned_images)}")
        
        return cleaned_images, cleaned_labels
    
    def remove_duplicates(self, images, labels, phash_threshold=8):
        """
        Remove exact and near-duplicate images from the dataset.

        Uses MD5 for exact matches and perceptual hash (pHash) for near-duplicates.
        Keeps the first occurrence of each image.

        Args:
            images: numpy array of images (uint8 or float32)
            labels: numpy array of labels (string)
            phash_threshold: max Hamming distance to consider near-duplicate

        Returns:
            Deduplicated images and labels arrays
        """
        print("Checking for duplicate images...")

        seen_md5: set[str] = set()
        seen_phashes: list[imagehash.ImageHash] = []
        keep_indices = []
        exact_count = 0
        near_count = 0

        for idx, img in enumerate(tqdm(images, desc="Deduplicating")):
            # Work on uint8 for hashing
            img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)

            md5 = hashlib.md5(img_uint8.tobytes()).hexdigest()
            if md5 in seen_md5:
                exact_count += 1
                continue

            ph = imagehash.phash(pil_img)
            if any((ph - existing) <= phash_threshold for existing in seen_phashes):
                near_count += 1
                continue

            seen_md5.add(md5)
            seen_phashes.append(ph)
            keep_indices.append(idx)

        removed = len(images) - len(keep_indices)
        print(f"  Exact duplicates removed : {exact_count}")
        print(f"  Near-duplicates removed  : {near_count}  (pHash threshold ≤ {phash_threshold})")
        print(f"  Images kept              : {len(keep_indices)} / {len(images)}")

        return images[keep_indices], labels[keep_indices]

    def remove_cross_split_leakage(self, X_train, y_train, X_val, y_val, X_test, y_test, phash_threshold=8):
        """
        Remove images from val and test that are near-duplicates of any train image.

        Builds a pHash index of the train set, then filters val and test.
        Dropped images are reported but not moved to train (to avoid distribution shift).

        Args:
            phash_threshold: max Hamming distance to flag as leakage (default 8)

        Returns:
            Cleaned X_train, y_train, X_val, y_val, X_test, y_test
        """
        print("\nChecking for cross-split leakage (train ↔ val/test)...")

        def to_phash(img: np.ndarray) -> imagehash.ImageHash:
            uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            return imagehash.phash(Image.fromarray(uint8))

        # Build pHash index from train
        print("  Building pHash index from train set...")
        train_phashes = [to_phash(img) for img in tqdm(X_train, desc="  Hashing train", leave=False)]

        def filter_split(X, y, name):
            keep = []
            leaked = 0
            for idx, img in enumerate(tqdm(X, desc=f"  Checking {name}", leave=False)):
                ph = to_phash(img)
                if any((ph - t) <= phash_threshold for t in train_phashes):
                    leaked += 1
                else:
                    keep.append(idx)
            print(f"  {name:5s}: {leaked} leaking images removed → {len(keep)} kept")
            return X[keep], y[keep]

        X_val,  y_val  = filter_split(X_val,  y_val,  "val")
        X_test, y_test = filter_split(X_test, y_test, "test")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def normalize_images(self, images):
        """
        Normalize pixel values to [0, 1] range
        
        Args:
            images: numpy array of images
            
        Returns:
            Normalized images
        """
        return images.astype('float32') / 255.0
    
    def encode_labels(self, labels):
        """
        Encode string labels to integers and one-hot vectors
        
        Args:
            labels: Array of string labels
            
        Returns:
            encoded_labels: Integer encoded labels
            onehot_labels: One-hot encoded labels
        """
        # Encode using config.CLASS_NAMES order (NOT alphabetical sort)
        # LabelEncoder.fit() sorts alphabetically which breaks model predictions
        # since the model was trained with config.CLASS_NAMES index order.
        class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        encoded_labels = np.array([class_to_idx[l] for l in labels], dtype=np.int32)
        self.label_encoder.fit(self.class_names)  # keep fitted for inverse_transform

        # One-hot encode
        onehot_labels = to_categorical(encoded_labels, num_classes=config.NUM_CLASSES)
        
        return encoded_labels, onehot_labels
    
    def split_data(self, images, labels):
        """
        Split data into train, validation, and test sets
        
        Args:
            images: numpy array of images
            labels: numpy array of labels
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Splitting data...")
        
        # Convert one-hot labels to integers for stratification (sklearn requires 1D)
        stratify_all = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels

        # First split: train + val, test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            images, labels,
            test_size=config.TEST_SPLIT,
            random_state=config.RANDOM_SEED,
            stratify=stratify_all
        )

        # Second split: train, val
        val_size_adjusted = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)
        stratify_train_val = np.argmax(y_train_val, axis=1) if len(y_train_val.shape) > 1 else y_train_val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size_adjusted,
            random_state=config.RANDOM_SEED,
            stratify=stratify_train_val
        )
        
        print(f"Train set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Save processed data to disk
        
        Args:
            X_train, X_val, X_test: Image arrays
            y_train, y_val, y_test: Label arrays
        """
        print("Saving processed data...")
        
        np.save(config.PROCESSED_DATA_DIR / 'X_train.npy', X_train)
        np.save(config.PROCESSED_DATA_DIR / 'X_val.npy', X_val)
        np.save(config.PROCESSED_DATA_DIR / 'X_test.npy', X_test)
        np.save(config.PROCESSED_DATA_DIR / 'y_train.npy', y_train)
        np.save(config.PROCESSED_DATA_DIR / 'y_val.npy', y_val)
        np.save(config.PROCESSED_DATA_DIR / 'y_test.npy', y_test)
        
        print(f"Data saved to {config.PROCESSED_DATA_DIR}")
    
    def load_processed_data(self):
        """
        Load previously processed data from disk
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Loading processed data...")
        
        X_train = np.load(config.PROCESSED_DATA_DIR / 'X_train.npy')
        X_val = np.load(config.PROCESSED_DATA_DIR / 'X_val.npy')
        X_test = np.load(config.PROCESSED_DATA_DIR / 'X_test.npy')
        y_train = np.load(config.PROCESSED_DATA_DIR / 'y_train.npy')
        y_val = np.load(config.PROCESSED_DATA_DIR / 'y_val.npy')
        y_test = np.load(config.PROCESSED_DATA_DIR / 'y_test.npy')
        
        print(f"Loaded {len(X_train)} training images")
        print(f"Loaded {len(X_val)} validation images")
        print(f"Loaded {len(X_test)} test images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_distribution(self, labels):
        """
        Get distribution of classes in dataset
        
        Args:
            labels: Array of labels (can be integer or one-hot encoded)
            
        Returns:
            DataFrame with class distribution
        """
        # Convert one-hot to integer if needed
        if len(labels.shape) > 1:
            labels = np.argmax(labels, axis=1)
        
        # Get class names
        class_labels = self.label_encoder.inverse_transform(labels)
        
        # Count occurrences
        unique, counts = np.unique(class_labels, return_counts=True)
        
        df = pd.DataFrame({
            'Class': unique,
            'Count': counts,
            'Percentage': (counts / len(labels) * 100).round(2)
        })
        
        return df.sort_values('Count', ascending=False)


def prepare_dataset(use_csv=False, csv_path=None):
    """
    Main function to prepare dataset
    
    Args:
        use_csv: Whether to load from CSV
        csv_path: Path to CSV file if use_csv=True
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Set random seed
    config.set_seed()
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load raw data
    if use_csv and csv_path:
        images, labels = loader.load_dataset_from_csv(csv_path)
    else:
        images, labels = loader.load_dataset_from_folders()
    
    # Clean corrupted images
    images, labels = loader.clean_corrupted_images(images, labels)

    # Remove duplicate images
    images, labels = loader.remove_duplicates(images, labels)

    # Normalize images
    images = loader.normalize_images(images)
    
    # Encode labels
    _, labels_onehot = loader.encode_labels(labels)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(images, labels_onehot)

    # Remove cross-split leakage (train images leaking into val/test)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.remove_cross_split_leakage(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Save processed data
    loader.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Print class distribution
    print("\nClass Distribution:")
    print(loader.get_class_distribution(y_train))
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Example usage
    prepare_dataset()
