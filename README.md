# MNIST Digit Recognition

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the training script:
```bash
python mnist_classifier.py
```

## Project Structure

- `mnist_classifier.py`: Main script containing the model architecture, training, and evaluation code
- `requirements.txt`: List of Python dependencies
- After running the script:
  - `mnist_model.pth`: Saved trained model
  - `accuracy_plot.png`: Plot showing the model's accuracy over training epochs
  - `data/`: Directory containing the MNIST dataset (automatically downloaded)

## Model Architecture

The CNN model consists of:
- 2 convolutional layers
- Max pooling
- Dropout for regularization
- 2 fully connected layers
- Log softmax output layer

## Performance

The model typically achieves ~98% accuracy on the test set after 10 epochs of training.
