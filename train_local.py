import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import logging
from typing import Dict, List, Tuple
import struct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MNISTDataset(Dataset):
    def __init__(self, images_file: str, labels_file: str, transform=None):
        self.transform = transform
        
        # Read images
        with open(images_file, 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8)
            self.images = self.images.reshape(size, rows, cols).astype(np.float32) / 255.0
        
        # Read labels
        with open(labels_file, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        logger.info(f"Loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        
        # Add channel dimension
        image = torch.tensor(image).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        return self.conv(x)

class MNISTModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # Model architecture
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        
        # Detailed logging
        if batch_idx % 50 == 0:  # Log every 50 batches
            logger.info(f"Training - Batch {batch_idx}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        
        # Detailed logging for validation
        if batch_idx % 20 == 0:  # Log more frequently during validation
            logger.info(f"Validation - Batch {batch_idx}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc, 'preds': preds, 'targets': y}

    def validation_epoch_end(self, outputs):
        # Aggregate validation metrics
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
        # Log epoch-level metrics
        logger.info(f"\nValidation Epoch End:")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info(f"Average Accuracy: {avg_acc:.4f}\n")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=20,
            steps_per_epoch=len(self.train_dataloader()),
            pct_start=0.2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

def main():
    try:
        logger.info("Starting MNIST training script...")
        logger.info("Using device: CPU")
        
        # Check if dataset exists
        data_dir = 'data'
        train_images = os.path.join(data_dir, 'train-images-idx3-ubyte')
        train_labels = os.path.join(data_dir, 'train-labels-idx1-ubyte')
        test_images = os.path.join(data_dir, 't10k-images-idx3-ubyte')
        test_labels = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
        
        if not all(os.path.exists(f) for f in [train_images, train_labels, test_images, test_labels]):
            logger.error("Dataset files not found. Please run download_dataset.py first.")
            return
        
        logger.info("Creating data transforms...")
        # Create transforms
        train_transform = nn.Sequential(
            nn.RandomRotation(15),
            nn.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1))
        )
        
        logger.info("Loading datasets...")
        # Create datasets
        train_dataset = MNISTDataset(train_images, train_labels, transform=train_transform)
        test_dataset = MNISTDataset(test_images, test_labels)
        
        logger.info("Creating data loaders...")
        # Create data loaders with reduced num_workers for CPU
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,  # Reduced batch size for CPU
            shuffle=True,
            num_workers=0,  # No multiprocessing for CPU
            pin_memory=False  # Disable pin_memory for CPU
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,  # Reduced batch size for CPU
            shuffle=False,
            num_workers=0,  # No multiprocessing for CPU
            pin_memory=False  # Disable pin_memory for CPU
        )
        
        logger.info("Initializing model...")
        # Initialize model and trainer
        model = MNISTModel(learning_rate=1e-3)
        
        callbacks = [
            ModelCheckpoint(
                monitor='val_acc',
                dirpath='checkpoints',
                filename='mnist-{epoch:02d}-{val_acc:.4f}',
                save_top_k=3,
                mode='max',
                verbose=True  # Add verbose output
            ),
            EarlyStopping(
                monitor='val_acc',
                patience=5,
                mode='max',
                verbose=True  # Add verbose output
            ),
            LearningRateMonitor(logging_interval='step')
        ]
        
        logger.info("Setting up trainer...")
        trainer = pl.Trainer(
            max_epochs=20,
            accelerator='cpu',  # Explicitly set to CPU
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=1,
            gradient_clip_val=0.5
        )
        
        # Train model
        logger.info("\n" + "="*50)
        logger.info("Starting training...")
        logger.info("="*50 + "\n")
        trainer.fit(model, train_loader, test_loader)
        
        # Save model
        logger.info("\nSaving models...")
        os.makedirs('checkpoints', exist_ok=True)
        model_path = 'checkpoints/mnist_model.pth'
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved PyTorch model to {model_path}")
        
        # Save TorchScript model
        model.eval()
        example_input = torch.randn(1, 1, 28, 28)
        traced_model = torch.jit.trace(model, example_input)
        script_path = 'checkpoints/mnist_model_jit.pt'
        torch.jit.save(traced_model, script_path)
        logger.info(f"Saved TorchScript model to {script_path}")
        
        logger.info("\n" + "="*50)
        logger.info("Training completed successfully!")
        logger.info("="*50)
        logger.info("\nModel files saved:")
        logger.info(f"1. {model_path}")
        logger.info(f"2. {script_path}")
        
    except Exception as e:
        logger.error(f"\nAn error occurred during training:")
        logger.error(f"{str(e)}")
        raise

if __name__ == '__main__':
    main()
