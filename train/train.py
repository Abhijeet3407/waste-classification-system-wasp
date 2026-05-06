"""
Training pipeline for waste classification model
Handles model training, validation, callbacks, and checkpointing
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.data_loader import DataLoader, prepare_dataset
from utils.augmentation import create_data_generators
from utils.model_builder import ModelBuilder


class ModelTrainer:
    """Handle model training operations"""
    
    def __init__(self, model=None, model_type=None):
        """
        Initialize ModelTrainer
        
        Args:
            model: Pre-built Keras model (optional)
            model_type: Type of model to build if model not provided
        """
        self.model_type = model_type or config.MODEL_TYPE
        self.model = model
        self.history = None
        self.callbacks = []
        
        # Create timestamp for this training run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.model_type}_{self.timestamp}"
    
    def setup_callbacks(self):
        """
        Setup training callbacks
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Model Checkpoint - save best model
        checkpoint_path = config.CHECKPOINTS_DIR / f"{self.run_name}_best.keras"
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=config.CHECKPOINT_MONITOR,
            mode=config.CHECKPOINT_MODE,
            save_best_only=config.SAVE_BEST_ONLY,
            verbose=1
        )
        callbacks.append(checkpoint)
        print(f"Checkpoint callback: Saving to {checkpoint_path}")
        
        # Early Stopping — monitor val_accuracy so a rising loss doesn't kill a learning model
        if config.USE_EARLY_STOPPING:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=config.EARLY_STOPPING_PATIENCE,
                min_delta=config.EARLY_STOPPING_MIN_DELTA,
                mode='max',
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
            print(f"Early stopping: patience={config.EARLY_STOPPING_PATIENCE}")
        
        # Learning Rate Scheduler
        if config.USE_LR_SCHEDULE:
            if config.LR_SCHEDULE_TYPE == 'reduce_on_plateau':
                lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=config.LR_FACTOR,
                    patience=config.LR_PATIENCE,
                    min_lr=config.LR_MIN,
                    verbose=1
                )
            elif config.LR_SCHEDULE_TYPE == 'exponential':
                def exponential_decay(epoch, lr):
                    return lr * 0.95
                lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay)
            else:
                lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=config.LR_FACTOR,
                    patience=config.LR_PATIENCE,
                    verbose=1
                )
            
            callbacks.append(lr_scheduler)
            print(f"Learning rate scheduler: {config.LR_SCHEDULE_TYPE}")
        
        # TensorBoard
        if config.TENSORBOARD_ENABLED:
            log_dir = config.LOGS_DIR / self.run_name
            tensorboard = keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
            callbacks.append(tensorboard)
            print(f"TensorBoard: Logs at {log_dir}")
        
        # CSV Logger
        csv_path = config.LOGS_DIR / f"{self.run_name}_training.csv"
        csv_logger = keras.callbacks.CSVLogger(
            str(csv_path),
            separator=',',
            append=False
        )
        callbacks.append(csv_logger)
        
        self.callbacks = callbacks
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of epochs (uses config if None)
            batch_size: Batch size (uses config if None)
            
        Returns:
            Training history
        """
        epochs = epochs or config.EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Model: {self.model_type}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Augmentation: {config.APPLY_AUGMENTATION}")
        print("="*60 + "\n")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Compute class weights to handle imbalanced datasets
        y_train_int = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
        class_weights_arr = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_int),
            y=y_train_int
        )
        class_weight_dict = dict(enumerate(class_weights_arr))
        print(f"Class weights: { {config.CLASS_NAMES[k]: round(v, 3) for k, v in class_weight_dict.items()} }")

        # Create data generators
        if config.APPLY_AUGMENTATION:
            train_generator, val_generator = create_data_generators(
                X_train, y_train, X_val, y_val, batch_size
            )

            # Train with generators
            self.history = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=config.VERBOSE
            )
        else:
            # Train without augmentation
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=config.VERBOSE
            )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        return self.history
    
    def plot_training_history(self, save_path=None):
        """
        Plot training and validation metrics
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.history is None:
            print("No training history available")
            return
        
        history = self.history.history
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {self.run_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy
        axes[0, 0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss
        axes[0, 1].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Train Precision', linewidth=2)
            axes[1, 0].plot(history['val_precision'], label='Val Precision', linewidth=2)
            axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Train Recall', linewidth=2)
            axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2)
            axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = config.RESULTS_DIR / f"{self.run_name}_history.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()
    
    def save_final_model(self, model_name=None):
        """
        Save the final trained model
        
        Args:
            model_name: Custom name for saved model
        """
        if model_name is None:
            model_name = f"{self.run_name}_final.keras"
        
        model_path = config.FINAL_MODELS_DIR / model_name
        self.model.save(str(model_path))
        print(f"Final model saved to {model_path}")
        
        return model_path


def main():
    """Main training function"""
    
    # Set random seed for reproducibility
    config.set_seed()
    
    # Print configuration
    config.print_config()
    
    # Load processed data (auto-prepare if not yet done)
    print("\nLoading data...")
    loader = DataLoader()

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = loader.load_processed_data()
    except FileNotFoundError:
        print("Processed data not found — running data preparation now...")
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset()
    
    # Build model
    print("\nBuilding model...")
    builder = ModelBuilder(model_type=config.MODEL_TYPE)
    model = builder.build_model()
    model = builder.compile_model(model)
    builder.print_model_summary(model)
    
    # Create trainer
    trainer = ModelTrainer(model=model, model_type=config.MODEL_TYPE)
    
    # Train model
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save final model
    model_path = trainer.save_final_model()
    
    # Print final metrics
    print("\n" + "="*60)
    print("FINAL TRAINING METRICS")
    print("="*60)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    # Check for overfitting
    overfit_gap = final_train_acc - final_val_acc
    if overfit_gap > 0.1:
        print(f"\n⚠️  Warning: Potential overfitting detected (gap: {overfit_gap:.4f})")
        print("Consider: more data augmentation, higher dropout, or regularization")
    
    print("="*60)
    print(f"\n✓ Training completed successfully!")
    print(f"✓ Model saved at: {model_path}")
    print(f"✓ Next step: Run evaluation with 'python evaluate/evaluate.py'")


if __name__ == "__main__":
    main()
