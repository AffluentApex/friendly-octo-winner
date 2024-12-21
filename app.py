from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import base64
import numpy as np
import cv2
from PIL import Image
import io
from models import BasicCNN, ResNet
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

models = {
    'basic_cnn': BasicCNN().to(device),
    'resnet': ResNet().to(device)
}

# Load trained models
try:
    for name, model in models.items():
        print(f"Loading {name} model...")
        model.load_state_dict(torch.load(f'checkpoints/{name}_best.pth'))
        model.eval()
        print(f"Successfully loaded {name} model")
except Exception as e:
    print(f"Error loading models: {e}")

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
        print(f"Error in preprocessing: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data and model choice from request
        data = request.json
        image_data = data['image']
        model_name = data.get('model', 'basic_cnn')  # default to basic_cnn
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 400
        
        # Preprocess image
        image_tensor = preprocess_image(image_data)
        if image_tensor is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            model = models[model_name]
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_probs, pred_class = torch.max(probabilities, 1)
            
            # Get all class probabilities
            all_probs = probabilities[0].tolist()
            
            prediction = {
                'digit': int(pred_class.item()),
                'confidence': float(pred_probs.item()),
                'probabilities': all_probs,
                'model_used': model_name
            }
            
            print(f"Prediction made: {prediction['digit']} with confidence: {prediction['confidence']:.2f}")
            return jsonify(prediction)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/webcam_predict', methods=['POST'])
def webcam_predict():
    try:
        # Get image data from request
        image_data = request.json['image']
        model_name = request.json.get('model', 'basic_cnn')  # default to basic_cnn
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 400
        
        # Convert base64 to OpenCV format
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(gray)
        
        # Apply transformations
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            model = models[model_name]
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_probs, pred_class = torch.max(probabilities, 1)
            
            prediction = {
                'digit': int(pred_class.item()),
                'confidence': float(pred_probs.item()),
                'probabilities': probabilities[0].tolist(),
                'model_used': model_name
            }
            
            print(f"Webcam prediction made: {prediction['digit']} with confidence: {prediction['confidence']:.2f}")
            return jsonify(prediction)
    
    except Exception as e:
        print(f"Error in webcam prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_available_models():
    return jsonify({'models': list(models.keys())})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
