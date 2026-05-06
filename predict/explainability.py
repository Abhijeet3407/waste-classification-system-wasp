"""
Model explainability using Grad-CAM
Visualize what the model is looking at when making predictions
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from predict.predict import WasteClassifier


class GradCAM:
    """Generate Grad-CAM visualizations for model interpretability"""
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            layer_name: Name of convolutional layer to use (auto-detect if None)
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        
        # Create grad model
        self.grad_model = self._create_grad_model()
    
    def _find_last_conv_layer(self):
        """Automatically find the last convolutional layer"""
        # Search through all layers
        for layer in reversed(self.model.layers):
            # Check if it's a Conv2D layer
            if hasattr(layer, 'layers'):
                # It's a nested model, search inside
                for inner_layer in reversed(layer.layers):
                    if isinstance(inner_layer, keras.layers.Conv2D):
                        return inner_layer.name
            elif isinstance(layer, keras.layers.Conv2D):
                return layer.name
        
        raise ValueError("No convolutional layer found in model")
    
    def _create_grad_model(self):
        """Create gradient model for Grad-CAM"""
        # Get the target layer
        target_layer = self.model.get_layer(self.layer_name)
        
        # Create model that maps input to layer output and final predictions
        grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[target_layer.output, self.model.output]
        )
        
        return grad_model
    
    def generate_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            img_array: Preprocessed image array (with batch dimension)
            pred_index: Class index to visualize (uses top prediction if None)
            
        Returns:
            Heatmap array
        """
        # Get gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap — epsilon prevents divide-by-zero when all activations are negative
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap
            original_img: Original image (PIL Image or numpy array)
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap to use
            
        Returns:
            Overlayed image
        """
        # Convert PIL to numpy if needed
        if hasattr(original_img, 'convert'):
            original_img = np.array(original_img)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is uint8
        if original_img.dtype == np.float32 or original_img.dtype == np.float64:
            original_img = np.uint8(255 * original_img)
        
        # Overlay
        overlayed = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed


class ModelExplainer:
    """High-level interface for model explainability"""
    
    def __init__(self, model_path=None):
        """
        Initialize ModelExplainer
        
        Args:
            model_path: Path to trained model
        """
        self.classifier = WasteClassifier(model_path)
        self.gradcam = GradCAM(self.classifier.model)
    
    def explain_prediction(self, image_path, save_path=None):
        """
        Generate complete explanation for a prediction
        
        Args:
            image_path: Path to image
            save_path: Path to save visualization
            
        Returns:
            Prediction result with visualizations
        """
        # Get prediction
        result = self.classifier.predict(image_path)
        
        # Preprocess image
        img_array, original_img = self.classifier.preprocess_image(image_path)
        
        # Generate Grad-CAM for top prediction
        heatmap = self.gradcam.generate_heatmap(img_array)
        overlayed = self.gradcam.overlay_heatmap(heatmap, original_img)
        
        # Create visualization
        self._visualize_explanation(
            original_img, overlayed, heatmap, result, save_path
        )
        
        return result
    
    def _visualize_explanation(self, original_img, overlayed, heatmap, result, save_path):
        """
        Create comprehensive visualization
        
        Args:
            original_img: Original image
            overlayed: Grad-CAM overlay
            heatmap: Raw heatmap
            result: Prediction result
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        
        # Grad-CAM overlay
        axes[0, 1].imshow(overlayed)
        axes[0, 1].axis('off')
        axes[0, 1].set_title(
            f'Grad-CAM: {result["predicted_class"]}\n'
            f'Confidence: {result["confidence_percentage"]:.2f}%',
            fontsize=12,
            fontweight='bold'
        )
        
        # Heatmap only
        axes[1, 0].imshow(heatmap, cmap='jet')
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Attention Heatmap', fontsize=12, fontweight='bold')
        
        # Prediction probabilities
        all_preds = result['all_predictions']
        classes = list(all_preds.keys())
        confidences = list(all_preds.values())
        
        colors = ['green' if c == result['predicted_class'] else 'skyblue' 
                  for c in classes]
        
        axes[1, 1].barh(classes, confidences, color=colors, alpha=0.8)
        axes[1, 1].set_xlabel('Confidence', fontsize=11)
        axes[1, 1].set_title('Class Probabilities', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        for i, (cls, conf) in enumerate(zip(classes, confidences)):
            axes[1, 1].text(conf + 0.02, i, f'{conf*100:.1f}%', 
                          va='center', fontsize=9)
        
        plt.suptitle(
            'Model Explainability: What is the model looking at?',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explanation saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_predictions(self, image_paths, save_path=None):
        """
        Compare Grad-CAM for multiple images
        
        Args:
            image_paths: List of image paths
            save_path: Path to save visualization
        """
        n_images = len(image_paths)
        fig, axes = plt.subplots(n_images, 3, figsize=(15, 5 * n_images))
        
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        for idx, img_path in enumerate(image_paths):
            # Get prediction and visualizations
            result = self.classifier.predict(img_path)
            img_array, original_img = self.classifier.preprocess_image(img_path)
            heatmap = self.gradcam.generate_heatmap(img_array)
            overlayed = self.gradcam.overlay_heatmap(heatmap, original_img)
            
            # Plot
            axes[idx, 0].imshow(original_img)
            axes[idx, 0].axis('off')
            axes[idx, 0].set_title(f'Image {idx+1}', fontsize=11, fontweight='bold')
            
            axes[idx, 1].imshow(overlayed)
            axes[idx, 1].axis('off')
            axes[idx, 1].set_title(
                f'{result["predicted_class"]} ({result["confidence_percentage"]:.1f}%)',
                fontsize=11,
                fontweight='bold'
            )
            
            axes[idx, 2].imshow(heatmap, cmap='jet')
            axes[idx, 2].axis('off')
            axes[idx, 2].set_title('Attention', fontsize=11, fontweight='bold')
        
        plt.suptitle('Grad-CAM Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main explainability function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Explainability with Grad-CAM')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--output', type=str, help='Output path for visualization')
    parser.add_argument('--model', type=str, help='Path to specific model file')
    
    args = parser.parse_args()
    
    # Create explainer
    explainer = ModelExplainer(args.model)
    
    # Generate explanation
    result = explainer.explain_prediction(args.image_path, args.output)
    
    # Print prediction
    explainer.classifier.print_prediction(result)
    
    print("\n" + "="*60)
    print("GRAD-CAM EXPLANATION")
    print("="*60)
    print("The heatmap shows which regions of the image the model")
    print("focused on when making its prediction.")
    print("Red/yellow areas = high attention")
    print("Blue/purple areas = low attention")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python predict/explainability.py <image_path> [--output <save_path>]")
        print("\nExample:")
        print("  python predict/explainability.py test_image.jpg")
        print("  python predict/explainability.py test_image.jpg --output explanation.png")
    else:
        main()
