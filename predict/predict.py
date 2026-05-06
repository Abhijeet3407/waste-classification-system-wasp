"""
Prediction script for waste classification
Classify new waste images and return predictions with confidence
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class WasteClassifier:
    """Classify waste images using trained model"""
    
    def __init__(self, model_path=None):
        """
        Initialize WasteClassifier
        
        Args:
            model_path: Path to trained model (uses latest if None)
        """
        self.class_names = config.CLASS_NAMES
        self.img_size = config.IMG_SIZE
        self.model_path = model_path
        self.model = None
        
        # Load model
        if model_path is None:
            self.model_path = self._find_latest_model()
        
        self.load_model()
    
    def _find_latest_model(self):
        """Find the most recent trained model"""
        model_files = list(config.FINAL_MODELS_DIR.glob("*.keras"))
        
        if not model_files:
            # Try checkpoints directory
            model_files = list(config.CHECKPOINTS_DIR.glob("*.keras"))
        
        if not model_files:
            raise FileNotFoundError(
                "No trained model found. Please train a model first.\n"
                "Run: python train/train.py"
            )
        
        # Get most recent model
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        return latest_model
    
    def load_model(self):
        """Load trained model and align class names to model output size"""
        print(f"Loading model: {self.model_path.name}")
        self.model = keras.models.load_model(self.model_path)
        # Derive number of classes from model output — guards against
        # config/model mismatches when a new model is still training
        num_classes = self.model.output_shape[-1]
        if num_classes != len(config.CLASS_NAMES):
            print(f"Note: model has {num_classes} outputs, config has {len(config.CLASS_NAMES)} classes — using first {num_classes}")
        self.class_names = config.CLASS_NAMES[:num_classes]
        print(f"Model loaded successfully ({num_classes} classes: {self.class_names})")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            Preprocessed image array
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            img = Image.open(image_path)
        else:
            img = image_path
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(self.img_size, Image.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    
    def predict(self, image_path, top_k=3):
        """
        Predict waste category for an image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Preprocess image
        img_array, original_img = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top K predictions — clamp to available classes
        top_k = min(top_k, len(predictions))
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_classes = [self.class_names[i] for i in top_indices]
        top_confidences = [float(predictions[i]) for i in top_indices]
        
        # Primary prediction
        predicted_class = top_classes[0]
        confidence = top_confidences[0]
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'confidence_percentage': confidence * 100,
            'top_predictions': list(zip(top_classes, top_confidences)),
            'all_predictions': {
                self.class_names[i]: float(predictions[i]) 
                for i in range(len(self.class_names))
            }
        }
        
        return result
    
    def predict_and_display(self, image_path, save_path=None):
        """
        Predict and display results with visualization
        
        Args:
            image_path: Path to image file
            save_path: Path to save visualization
        """
        # Get prediction
        result = self.predict(image_path)
        
        # Load original image
        _, original_img = self.preprocess_image(image_path)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Display image
        ax1.imshow(original_img)
        ax1.axis('off')
        ax1.set_title(
            f"Predicted: {result['predicted_class']}\n"
            f"Confidence: {result['confidence_percentage']:.2f}%",
            fontsize=14,
            fontweight='bold'
        )
        
        # Display prediction bars
        all_preds = result['all_predictions']
        classes = list(all_preds.keys())
        confidences = list(all_preds.values())
        
        colors = ['green' if c == result['predicted_class'] else 'skyblue' 
                  for c in classes]
        
        ax2.barh(classes, confidences, color=colors, alpha=0.8)
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
        ax2.set_xlim([0, 1])
        ax2.grid(axis='x', alpha=0.3)
        
        # Add confidence values on bars
        for i, (cls, conf) in enumerate(zip(classes, confidences)):
            ax2.text(conf + 0.02, i, f'{conf*100:.1f}%', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result
    
    def print_prediction(self, result):
        """
        Print prediction results in a formatted way
        
        Args:
            result: Prediction result dictionary
        """
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Predicted Class: {result['predicted_class'].upper()}")
        print(f"Confidence: {result['confidence_percentage']:.2f}%")
        print("\n" + "-"*60)
        print("Top Predictions:")
        print("-"*60)
        
        for i, (cls, conf) in enumerate(result['top_predictions'], 1):
            bar_length = int(conf * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"{i}. {cls:12s} {bar} {conf*100:5.2f}%")
        
        print("="*60 + "\n")


def predict_single_image(image_path, visualize=True):
    """
    Convenience function to predict a single image
    
    Args:
        image_path: Path to image
        visualize: Whether to show visualization
        
    Returns:
        Prediction result dictionary
    """
    classifier = WasteClassifier()
    
    if visualize:
        result = classifier.predict_and_display(image_path)
    else:
        result = classifier.predict(image_path)
    
    classifier.print_prediction(result)
    
    return result


def predict_batch(image_folder, output_csv=None):
    """
    Predict multiple images in a folder

    Args:
        image_folder: Path to folder containing images
        output_csv: Path to save results CSV

    Returns:
        DataFrame with all predictions
    """
    classifier = WasteClassifier()
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_folder = Path(image_folder)
    image_files = [
        f for f in image_folder.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_files)} images to classify")
    
    # Predict all images
    results = []
    
    for img_file in image_files:
        try:
            result = classifier.predict(img_file)
            results.append({
                'filename': img_file.name,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence_percentage']
            })
            print(f"✓ {img_file.name}: {result['predicted_class']} ({result['confidence_percentage']:.2f}%)")
        except Exception as e:
            print(f"✗ Error processing {img_file.name}: {e}")
            results.append({
                'filename': img_file.name,
                'predicted_class': 'ERROR',
                'confidence': 0.0
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
    
    return df


def main():
    """Main prediction function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Waste Classification Prediction')
    parser.add_argument('image_path', type=str, help='Path to image file or folder')
    parser.add_argument('--batch', action='store_true', help='Process folder of images')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--output', type=str, help='Output CSV path (for batch mode)')
    parser.add_argument('--model', type=str, help='Path to specific model file')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch prediction
        predict_batch(args.image_path, args.output)
    else:
        # Single image prediction
        predict_single_image(args.image_path, visualize=not args.no_viz)


if __name__ == "__main__":
    # If no arguments provided, show usage
    if len(sys.argv) == 1:
        print("Usage:")
        print("  Single image: python predict/predict.py <image_path>")
        print("  Batch mode:   python predict/predict.py <folder_path> --batch")
        print("\nExample:")
        print("  python predict/predict.py test_image.jpg")
        print("  python predict/predict.py test_images/ --batch --output results.csv")
    else:
        main()
