# MNIST Digit Recognition Web App

A Flask-based web application for real-time handwritten digit recognition using deep learning.

## Overview

This project implements a handwritten digit recognition system using PyTorch and Flask. The application features a modern, glass-morphism UI design that allows users to draw digits and get real-time predictions with confidence scores.

## Model Training

Initially, we trained the model locally, but due to lower confidence scores in predictions, we moved to Google Colab with TPU acceleration for improved performance. The TPU training code can be found in `train_tpu.py`, which implements:

- Enhanced model architecture with residual connections
- TPU-optimized training loop
- Data augmentation for better generalization
- Learning rate scheduling
- Validation monitoring

The TPU-trained model achieved significantly better accuracy and confidence in predictions compared to our locally trained models.

## Features

- Real-time digit recognition
- Modern glass-morphism UI design
- Confidence scores for predictions
- Probability distribution for all digits
- Mobile-responsive design
- Touch screen support

## Setup and Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download the trained model (not included in repository due to size)
4. Place the model in the `models` directory
5. Run the Flask app:
```bash
python app.py
```

## Model Architecture

We use an enhanced CNN architecture (`EnhancedMNISTNet`) with the following improvements:
- Residual connections
- Batch normalization
- Dropout for regularization
- Increased network depth

## Training Details

The final model was trained on Google Colab using TPU acceleration with the following specifications:
- Training epochs: 50
- Batch size: 1024 (optimized for TPU)
- Learning rate: 0.001 with cosine annealing
- Data augmentation: Random rotation, shifts, and zoom
- Validation split: 20%

## Usage

1. Open the web application in your browser
2. Draw a digit using your mouse or touch screen
3. Click "Recognize" to get the prediction
4. View the confidence score and probability distribution
5. Use "Clear" to reset the canvas

## Files

- `app.py`: Flask application
- `models.py`: Neural network architecture
- `train_tpu.py`: TPU-accelerated training code
- `templates/index.html`: Frontend UI
- `models/`: Directory containing trained models
- `download_dataset.py`: MNIST dataset downloader

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License
