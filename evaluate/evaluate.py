"""
Model evaluation script
Comprehensive evaluation with metrics, confusion matrix, and visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.data_loader import DataLoader


class ModelEvaluator:
    """Evaluate trained model performance"""
    
    def __init__(self, model_path):
        """
        Initialize ModelEvaluator
        
        Args:
            model_path: Path to saved model
        """
        self.model_path = model_path
        self.model = None
        self.class_names = config.CLASS_NAMES
        self.load_model()
    
    def load_model(self):
        """Load trained model from disk"""
        print(f"Loading model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        print("Model loaded successfully")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        # Get predictions
        print("Generating predictions...")
        y_pred_probs = self.model.predict(X_test, batch_size=32, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class metrics — zero_division=0 prevents warnings when a class has no predictions
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs
        }
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Test Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"F1-Score:       {f1:.4f}")
        print("="*60)
        
        return results
    
    def print_classification_report(self, y_true, y_pred):
        """
        Print detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )
        print(report)
        
        # Save report to file
        report_path = config.RESULTS_DIR / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {report_path}")
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'},
            square=True
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = config.RESULTS_DIR / "confusion_matrix.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_per_class_metrics(self, results, save_path=None):
        """
        Plot per-class performance metrics
        
        Args:
            results: Dictionary of evaluation results
            save_path: Path to save plot
        """
        # Prepare data
        metrics_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': results['precision_per_class'],
            'Recall': results['recall_per_class'],
            'F1-Score': results['f1_per_class']
        })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set bar positions
        x = np.arange(len(self.class_names))
        width = 0.25
        
        # Plot bars
        ax.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = config.RESULTS_DIR / "per_class_metrics.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to {save_path}")
        plt.close()
        
        # Print table
        print("\n" + "="*60)
        print("PER-CLASS PERFORMANCE")
        print("="*60)
        print(metrics_df.to_string(index=False, float_format='%.4f'))
        print("="*60)
    
    def plot_prediction_confidence(self, y_pred_probs, y_true, save_path=None):
        """
        Plot prediction confidence distribution
        
        Args:
            y_pred_probs: Predicted probabilities
            y_true: True labels
            save_path: Path to save plot
        """
        # Get max confidence for each prediction
        confidences = np.max(y_pred_probs, axis=1)
        
        # Separate correct and incorrect predictions
        y_pred = np.argmax(y_pred_probs, axis=1)
        correct_mask = (y_pred == y_true)
        
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Histogram
        axes[0].hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green')
        axes[0].hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red')
        axes[0].set_xlabel('Confidence', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Box plot
        data_to_plot = [correct_confidences, incorrect_confidences]
        bp = axes[1].boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        axes[1].set_ylabel('Confidence', fontsize=12)
        axes[1].set_title('Confidence: Correct vs Incorrect', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = config.RESULTS_DIR / "prediction_confidence.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction confidence plot saved to {save_path}")
        plt.close()
        
        # Print statistics
        print("\n" + "="*60)
        print("PREDICTION CONFIDENCE STATISTICS")
        print("="*60)
        print(f"Correct predictions - Mean confidence: {np.mean(correct_confidences):.4f}")
        print(f"Correct predictions - Median confidence: {np.median(correct_confidences):.4f}")
        print(f"Incorrect predictions - Mean confidence: {np.mean(incorrect_confidences):.4f}")
        print(f"Incorrect predictions - Median confidence: {np.median(incorrect_confidences):.4f}")
        print("="*60)
    
    def save_evaluation_summary(self, results):
        """
        Save evaluation summary to CSV
        
        Args:
            results: Dictionary of evaluation results
        """
        summary = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [
                results['accuracy'],
                results['precision'],
                results['recall'],
                results['f1_score']
            ]
        }
        
        df = pd.DataFrame(summary)
        
        summary_path = config.RESULTS_DIR / "evaluation_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"Evaluation summary saved to {summary_path}")


def main():
    """Main evaluation function"""
    
    # Check if model exists
    model_files = list(config.FINAL_MODELS_DIR.glob("*.keras"))
    
    if not model_files:
        print("No trained models found!")
        print(f"Please train a model first: python train/train.py")
        return
    
    # Use most recent model
    model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"Using model: {model_path.name}")
    
    # Load test data
    print("\nLoading test data...")
    loader = DataLoader()
    
    try:
        _, _, X_test, _, _, y_test = loader.load_processed_data()
    except FileNotFoundError:
        print("Processed data not found. Please run data preparation first.")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path)
    
    # Evaluate model
    results = evaluator.evaluate(X_test, y_test)
    
    # Generate reports and plots
    evaluator.print_classification_report(results['y_true'], results['y_pred'])
    evaluator.plot_confusion_matrix(results['y_true'], results['y_pred'])
    evaluator.plot_per_class_metrics(results)
    evaluator.plot_prediction_confidence(results['y_pred_probs'], results['y_true'])
    evaluator.save_evaluation_summary(results)
    
    print("\n" + "="*60)
    print("✓ Evaluation completed successfully!")
    print(f"✓ Results saved in: {config.RESULTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
