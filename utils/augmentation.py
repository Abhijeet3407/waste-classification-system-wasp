"""
Data augmentation utilities
Implements various augmentation techniques for training
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
import config


class DataAugmentor:
    """Handle data augmentation operations"""
    
    def __init__(self, augmentation_params=None):
        """
        Initialize DataAugmentor
        
        Args:
            augmentation_params: Dictionary of augmentation parameters
        """
        self.params = augmentation_params or config.AUGMENTATION_PARAMS
    
    def create_keras_generator(self):
        """
        Create Keras ImageDataGenerator for augmentation
        
        Returns:
            ImageDataGenerator instance
        """
        datagen = ImageDataGenerator(
            rotation_range=self.params.get('rotation_range', 20),
            width_shift_range=self.params.get('width_shift_range', 0.2),
            height_shift_range=self.params.get('height_shift_range', 0.2),
            horizontal_flip=self.params.get('horizontal_flip', True),
            vertical_flip=self.params.get('vertical_flip', False),
            zoom_range=self.params.get('zoom_range', 0.15),
            brightness_range=self.params.get('brightness_range', [0.8, 1.2]),
            fill_mode=self.params.get('fill_mode', 'nearest'),
            shear_range=self.params.get('shear_range', 10)
        )
        
        return datagen
    
    def create_albumentations_transform(self):
        """
        Create Albumentations transformation pipeline
        More advanced augmentation techniques
        
        Returns:
            Albumentations Compose object
        """
        transform = A.Compose([
            A.Rotate(limit=20, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=20,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=5),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.05),
                A.GridDistortion(num_steps=5, distort_limit=0.05),
            ], p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                fill_value=0,
                p=0.3
            ),
        ])
        
        return transform
    
    def augment_with_keras(self, X_train, y_train, batch_size=32):
        """
        Create augmented data generator using Keras
        
        Args:
            X_train: Training images
            y_train: Training labels
            batch_size: Batch size for generator
            
        Returns:
            Data generator
        """
        datagen = self.create_keras_generator()
        datagen.fit(X_train)
        
        generator = datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        return generator
    
    def augment_with_albumentations(self, image):
        """
        Apply Albumentations transform to a single image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Augmented image
        """
        transform = self.create_albumentations_transform()
        
        # Albumentations expects uint8 images
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        augmented = transform(image=image)
        augmented_image = augmented['image']
        
        # Convert back to float32 [0, 1]
        augmented_image = augmented_image.astype(np.float32) / 255.0
        
        return augmented_image
    
    def visualize_augmentation(self, image, num_augmentations=9):
        """
        Visualize augmentation results
        
        Args:
            image: Input image
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented images
        """
        augmented_images = [image]  # Original image
        
        for _ in range(num_augmentations - 1):
            aug_img = self.augment_with_albumentations(image)
            augmented_images.append(aug_img)
        
        return augmented_images


def create_data_generators(X_train, y_train, X_val, y_val, batch_size=None):
    """
    Create training and validation data generators
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size (uses config if None)
        
    Returns:
        train_generator, val_generator
    """
    batch_size = batch_size or config.BATCH_SIZE
    
    if config.APPLY_AUGMENTATION:
        # Training generator with augmentation
        augmentor = DataAugmentor()
        train_generator = augmentor.augment_with_keras(X_train, y_train, batch_size)
        
        # Validation generator without augmentation
        val_datagen = ImageDataGenerator()
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
        
        print(f"Created augmented data generators")
        print(f"Training batches per epoch: {len(train_generator)}")
        print(f"Validation batches: {len(val_generator)}")
        
        return train_generator, val_generator
    else:
        # No augmentation - simple generators
        train_datagen = ImageDataGenerator()
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
        
        return train_generator, val_generator


if __name__ == "__main__":
    # Example usage
    print("Data Augmentation Module")
    print("Augmentation parameters:", config.AUGMENTATION_PARAMS)
