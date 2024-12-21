import torch
from PIL import Image
import torchvision.transforms as transforms
from mnist_classifier import DigitClassifier
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_image(image_path):
    # Load image and convert to grayscale
    image = Image.open(image_path).convert('L')
    
    # Resize to 28x28 (MNIST size)
    image = image.resize((28, 28))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_digit(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True)
        probability = torch.exp(output).max()
    return prediction.item(), probability.item()

def display_prediction(image_path, prediction, probability):
    # Load and display the original image
    image = Image.open(image_path).convert('L')
    plt.figure(figsize=(6, 4))
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted Digit: {prediction}\nConfidence: {probability:.2%}')
    plt.axis('off')
    plt.show()

def main():
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitClassifier().to(device)
    
    try:
        model.load_state_dict(torch.load('mnist_model.pth'))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Trained model file 'mnist_model.pth' not found!")
        return

    # Create a test_images directory if it doesn't exist
    import os
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
        print("Created 'test_images' directory. Please place your test images there.")
        return

    # Process all images in the test_images directory
    test_dir = 'test_images'
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    found_images = False
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(valid_extensions):
            found_images = True
            image_path = os.path.join(test_dir, filename)
            print(f"\nProcessing image: {filename}")
            
            # Load and preprocess the image
            image_tensor = load_and_preprocess_image(image_path)
            image_tensor = image_tensor.to(device)
            
            # Make prediction
            prediction, probability = predict_digit(model, image_tensor)
            print(f"Predicted digit: {prediction}")
            print(f"Confidence: {probability:.2%}")
            
            # Display the image with prediction
            display_prediction(image_path, prediction, probability)
    
    if not found_images:
        print("\nNo valid images found in 'test_images' directory!")
        print("Please add some images (PNG, JPG, JPEG, or BMP) to test.")

if __name__ == '__main__':
    main()
