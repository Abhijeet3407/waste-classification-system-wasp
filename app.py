"""
Flask API for waste classification web application
Handles image uploads, predictions, and model inference
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from predict.predict import WasteClassifier
from predict.explainability import GradCAM
import matplotlib.pyplot as plt
import cv2

# Resolve frontend directory relative to this file (works regardless of cwd)
BASE_DIR     = Path(__file__).parent.absolute()
FRONTEND_DIR = BASE_DIR / 'frontend'

# Initialize Flask app — serve everything from frontend/
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
CORS(app)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE

# Initialize classifier (lazy loading)
classifier = None


def get_classifier():
    """Get or initialize classifier (singleton pattern)"""
    global classifier
    if classifier is None:
        print("Initializing waste classifier...")
        classifier = WasteClassifier()
        print("Classifier ready!")
    return classifier


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory(str(FRONTEND_DIR), 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'classes': config.CLASS_NAMES
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict waste category from uploaded image
    
    Expected: multipart/form-data with 'image' file
    Returns: JSON with prediction results
    """
    # Check if image is in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(filepath)
        
        # Get classifier
        clf = get_classifier()
        
        # Make prediction
        result = clf.predict(filepath)
        
        # Clean up uploaded file (optional)
        # filepath.unlink()
        
        # Return results
        return jsonify({
            'success': True,
            'prediction': {
                'class': result['predicted_class'],
                'confidence': result['confidence_percentage'],
                'top_predictions': [
                    {'class': cls, 'confidence': conf * 100}
                    for cls, conf in result['top_predictions']
                ],
                'all_predictions': {
                    cls: conf * 100 
                    for cls, conf in result['all_predictions'].items()
                }
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict_base64', methods=['POST'])
def predict_base64():
    """
    Predict from base64 encoded image
    
    Expected: JSON with 'image' field containing base64 string
    Returns: JSON with prediction results
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode and open image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Get classifier
        clf = get_classifier()
        
        # Make prediction
        result = clf.predict(image)
        
        # Return results
        return jsonify({
            'success': True,
            'prediction': {
                'class': result['predicted_class'],
                'confidence': result['confidence_percentage'],
                'top_predictions': [
                    {'class': cls, 'confidence': conf * 100}
                    for cls, conf in result['top_predictions']
                ],
                'all_predictions': {
                    cls: conf * 100 
                    for cls, conf in result['all_predictions'].items()
                }
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/explain', methods=['POST'])
def explain():
    """
    Generate Grad-CAM explanation for prediction
    
    Expected: multipart/form-data with 'image' file
    Returns: JSON with prediction and base64 encoded heatmap
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(filepath)
        
        # Get classifier
        clf = get_classifier()
        
        # Make prediction
        result = clf.predict(filepath)
        
        # Generate Grad-CAM
        img_array, original_img = clf.preprocess_image(filepath)
        gradcam = GradCAM(clf.model)
        heatmap = gradcam.generate_heatmap(img_array)
        overlayed = gradcam.overlay_heatmap(heatmap, original_img)
        
        # Convert overlayed image to base64
        overlayed_pil = Image.fromarray(overlayed.astype(np.uint8))
        buffered = BytesIO()
        overlayed_pil.save(buffered, format="PNG")
        overlayed_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Return results
        return jsonify({
            'success': True,
            'prediction': {
                'class': result['predicted_class'],
                'confidence': result['confidence_percentage'],
                'top_predictions': [
                    {'class': cls, 'confidence': conf * 100}
                    for cls, conf in result['top_predictions']
                ]
            },
            'gradcam_image': f'data:image/png;base64,{overlayed_base64}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        clf = get_classifier()
        
        return jsonify({
            'success': True,
            'model': {
                'name': clf.model_path.name,
                'classes': clf.class_names,
                'num_classes': len(clf.class_names),
                'input_size': clf.img_size,
                'model_type': config.MODEL_TYPE
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'max_size_mb': config.MAX_UPLOAD_SIZE / (1024 * 1024)
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Run the Flask application"""
    print("\n" + "="*60)
    print("WASTE CLASSIFICATION API SERVER")
    print("="*60)
    print(f"Host: {config.API_HOST}")
    print(f"Port: {config.API_PORT}")
    print(f"Upload folder: {config.UPLOAD_FOLDER}")
    print("="*60 + "\n")
    
    # Pre-load model
    print("Pre-loading model...")
    get_classifier()
    
    print("\n" + "="*60)
    print("Server ready! Access the app at:")
    print(f"http://localhost:{config.API_PORT}")
    print("="*60 + "\n")
    
    # Run server
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=True
    )


if __name__ == '__main__':
    main()
