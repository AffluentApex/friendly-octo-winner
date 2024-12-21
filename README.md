# MNIST Digit Recognition Web App

A sophisticated web application for real-time digit recognition using multiple deep learning models. The application provides both drawing pad and webcam interfaces for digit recognition.

## Features

### Multiple Model Architectures
- **Basic CNN**: A simple yet effective convolutional neural network
- **ResNet**: Deep residual network architecture
- **VGG-Style Network**: (Coming soon) Advanced VGG-style architecture

### Interactive Web Interface
- Drawing pad for digit input
- Real-time webcam digit recognition
- Model selection dropdown
- Confidence visualization with progress bar
- Clear and intuitive user interface

### Advanced Training Features
- Cross-validation support
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Data augmentation
- TensorBoard integration for visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mnist_digit_recognition.git
cd mnist_digit_recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the models:
```bash
python train_models.py
```
This will train all available models and save them in the `checkpoints` directory.

2. Start the web application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
mnist_digit_recognition/
├── app.py                 # Flask web application
├── models.py             # Neural network model architectures
├── train_models.py       # Training script
├── training_utils.py     # Training utilities and data transforms
├── test_custom_images.py # Testing script for custom images
├── requirements.txt      # Project dependencies
├── checkpoints/         # Saved model weights
├── data/               # MNIST dataset (downloaded automatically)
└── templates/          # HTML templates
    └── index.html      # Main web interface
```

## Model Architectures

### Basic CNN
- 3 convolutional layers
- 2 fully connected layers
- ReLU activation
- Max pooling
- Dropout for regularization

### ResNet
- Deep residual network
- Skip connections
- Batch normalization
- Advanced architecture for better accuracy

### VGG-Style Network (Coming Soon)
- Deep VGG-style architecture
- Multiple convolutional layers
- Advanced feature extraction

## Dependencies

- PyTorch
- Flask
- OpenCV
- NumPy
- TensorBoard
- scikit-learn
- tqdm

## Performance

Current model accuracies on MNIST test set:
- Basic CNN: ~98%
- ResNet: ~99%
- VGG (Coming soon)

## Contributing

Feel free to open issues or submit pull requests. All contributions are welcome!

## License

MIT License - feel free to use this project for your own learning and development.

## Acknowledgments

- MNIST Dataset
- PyTorch Team
- Flask Framework
