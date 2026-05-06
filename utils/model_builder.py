"""
Model building utilities
Contains implementations of different CNN architectures
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, MobileNetV2, EfficientNetB0, VGG16
)
import config


class ModelBuilder:
    """Build and configure different model architectures"""
    
    def __init__(self, model_type=None, input_shape=None, num_classes=None):
        """
        Initialize ModelBuilder
        
        Args:
            model_type: Type of model to build
            input_shape: Input image shape
            num_classes: Number of output classes
        """
        self.model_type = model_type or config.MODEL_TYPE
        self.input_shape = input_shape or config.INPUT_SHAPE
        self.num_classes = num_classes or config.NUM_CLASSES
    
    def build_model(self):
        """
        Build model based on specified type
        
        Returns:
            Compiled Keras model
        """
        print(f"Building {self.model_type} model...")
        
        if self.model_type == 'custom_cnn':
            model = self.build_custom_cnn()
        elif self.model_type == 'resnet50':
            model = self.build_resnet50()
        elif self.model_type == 'mobilenetv2':
            model = self.build_mobilenetv2()
        elif self.model_type == 'efficientnetb0':
            model = self.build_efficientnetb0()
        elif self.model_type == 'vgg16':
            model = self.build_vgg16()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Model built successfully: {self.model_type}")
        return model
    
    def build_custom_cnn(self):
        """
        Build custom CNN architecture from scratch
        Good baseline model for comparison
        
        Returns:
            Keras model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers — L2 regularization reduces overfitting on small datasets
            layers.Flatten(),
            layers.Dense(512, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_resnet50(self):
        """
        Build ResNet50 with transfer learning.
        ResNet50 preprocess_input expects zero-centred BGR from [0,255].
        Our data is [0,1] RGB → rescale to [0,255] then apply preprocess_input.
        """
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        if config.FREEZE_BASE_MODEL:
            base_model.trainable = False

        inputs = keras.Input(shape=self.input_shape)
        # [0,1] → [0,255] → ResNet50 zero-centering
        x = layers.Rescaling(scale=255.0)(inputs)
        x = keras.applications.resnet50.preprocess_input(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return keras.Model(inputs, outputs, name='resnet50_classifier')

    def build_mobilenetv2(self):
        """
        Build MobileNetV2 with transfer learning.
        MobileNetV2 preprocess_input maps [0,255] → [-1,1].
        Our data is [0,1] → Rescaling(2.0, -1.0) gives [-1,1] directly.
        """
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        if config.FREEZE_BASE_MODEL:
            base_model.trainable = False

        inputs = keras.Input(shape=self.input_shape)
        # [0,1] → [-1,1] as MobileNetV2 expects
        x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return keras.Model(inputs, outputs, name='mobilenetv2_classifier')

    def build_efficientnetb0(self):
        """
        Build EfficientNetB0 with transfer learning.
        EfficientNetB0 handles its own internal normalisation when given [0,255].
        Our data is [0,1] → rescale to [0,255] first.
        """
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        if config.FREEZE_BASE_MODEL:
            base_model.trainable = False

        inputs = keras.Input(shape=self.input_shape)
        # [0,1] → [0,255] for EfficientNet's internal normalisation
        x = layers.Rescaling(scale=255.0)(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return keras.Model(inputs, outputs, name='efficientnetb0_classifier')

    def build_vgg16(self):
        """
        Build VGG16 with transfer learning.
        VGG16 preprocess_input expects zero-centred BGR from [0,255].
        Our data is [0,1] → rescale then apply preprocess_input.
        """
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        if config.FREEZE_BASE_MODEL:
            base_model.trainable = False

        inputs = keras.Input(shape=self.input_shape)
        x = layers.Rescaling(scale=255.0)(inputs)
        x = keras.applications.vgg16.preprocess_input(x)
        x = base_model(x, training=False)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return keras.Model(inputs, outputs, name='vgg16_classifier')
    
    def compile_model(self, model, learning_rate=None):
        """
        Compile model with optimizer, loss, and metrics
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate (uses config if None)
            
        Returns:
            Compiled model
        """
        learning_rate = learning_rate or config.INITIAL_LEARNING_RATE
        
        # Choose optimizer
        if config.OPTIMIZER == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif config.OPTIMIZER == 'sgd':
            optimizer = keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9,
                nesterov=True
            )
        elif config.OPTIMIZER == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        print(f"Model compiled with {config.OPTIMIZER} optimizer")
        return model
    
    def print_model_summary(self, model):
        """
        Print model architecture summary
        
        Args:
            model: Keras model
        """
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*60)
        model.summary()
        print("="*60)
        
        # Count parameters using the built-in method
        trainable_params = sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        )
        non_trainable_params = sum(
            tf.keras.backend.count_params(w) for w in model.non_trainable_weights
        )
        total_params = trainable_params + non_trainable_params

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print("="*60 + "\n")
    
    def unfreeze_for_fine_tuning(self, model, num_layers=None):
        """
        Unfreeze layers for fine-tuning
        
        Args:
            model: Keras model
            num_layers: Number of layers to unfreeze from the end
            
        Returns:
            Model with unfrozen layers
        """
        num_layers = num_layers or config.FINE_TUNE_LAYERS
        
        # Get base model (first layer for Sequential models)
        if isinstance(model.layers[0], keras.Model):
            base_model = model.layers[0]
            
            # Unfreeze base model
            base_model.trainable = True
            
            # Freeze all layers except last num_layers
            for layer in base_model.layers[:-num_layers]:
                layer.trainable = False
            
            print(f"Unfroze last {num_layers} layers for fine-tuning")
        else:
            print("Model doesn't have a base model to fine-tune")
        
        return model


def create_model(model_type=None):
    """
    Convenience function to create and compile a model
    
    Args:
        model_type: Type of model to build
        
    Returns:
        Compiled Keras model
    """
    builder = ModelBuilder(model_type=model_type)
    model = builder.build_model()
    model = builder.compile_model(model)
    builder.print_model_summary(model)
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Model Builder Module")
    model = create_model()
