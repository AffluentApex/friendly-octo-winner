from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import base64
import numpy as np
from PIL import Image
import io
from models import EnhancedMNISTNet
import torchvision.transforms as transforms
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize model
model = EnhancedMNISTNet().to(device)

# Load the trained model
MODEL_PATH = r"C:\Users\Ronit\Downloads\best_mnist_model.pth"
logger.info(f"Loading model from: {MODEL_PATH}")

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Log checkpoint contents for debugging
    logger.info(f"Checkpoint keys: {checkpoint.keys()}")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        accuracy = checkpoint.get('accuracy', 'N/A')
        logger.info(f"Model loaded successfully with accuracy: {accuracy}%")
    else:
        # Try loading as direct state dict
        model.load_state_dict(checkpoint)
        logger.info("Model loaded successfully (direct state dict)")
    
    model.eval()
    logger.info("Model set to evaluation mode")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def preprocess_image(image_data):
    """Convert base64 image to tensor"""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data
        data = request.json
        image_data = data['image']
        
        # Preprocess image
        image_tensor = preprocess_image(image_data)
        if image_tensor is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_probs, pred_class = torch.max(probabilities, 1)
            
            # Get all class probabilities
            all_probs = probabilities[0].tolist()
            
            prediction = {
                'digit': int(pred_class.item()),
                'confidence': float(pred_probs.item()),
                'probabilities': all_probs
            }
            
            logger.info(f"Prediction made: {prediction['digit']} with confidence: {prediction['confidence']:.2f}")
            return jsonify(prediction)
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True)
