import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import logging
from typing import Dict, List, Tuple
import struct

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MNISTDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str, transform=None):
        self.transform = transform
        
        # Read images
        with open(images_path, 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8)
            self.images = self.images.reshape(size, rows, cols)
            
        # Read labels
        with open(labels_path, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
            
        logger.info(f"Loaded {len(self.images)} images of shape {self.images[0].shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        label = int(self.labels[idx])
        
        if self.transform:
            image = torch.tensor(image).unsqueeze(0)  # Add channel dimension
            image = self.transform(image)
            
        return image, label

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.bn(self.conv(x))))

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AdvancedMNISTModel(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # Feature extraction
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 32),
            SEBlock(32),
            nn.MaxPool2d(2),
            
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            SEBlock(64),
            nn.MaxPool2d(2),
            
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            SEBlock(128),
            nn.MaxPool2d(2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        
        # Metrics
        self.train_acc = pl.metrics.Accuracy(task='multiclass', num_classes=10)
        self.val_acc = pl.metrics.Accuracy(task='multiclass', num_classes=10)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                     batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.train_acc(logits.softmax(dim=-1), y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                       batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.val_acc(logits.softmax(dim=-1), y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=30,
            steps_per_epoch=len(self.train_dataloader()),
            pct_start=0.3,
            div_factor=10,
            final_div_factor=100
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def main():
    try:
        logger.info("Setting up...")
        
        # Download dataset
        path = kagglehub.dataset_download("hojjatk/mnist-dataset")
        logger.info(f"Dataset downloaded to: {path}")
        
        # Create transforms
        train_transform = nn.Sequential(
            nn.RandomRotation(15),
            nn.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            nn.Normalize((0.1307,), (0.3081,))
        )
        
        test_transform = nn.Sequential(
            nn.Normalize((0.1307,), (0.3081,))
        )
        
        # Create datasets
        train_dataset = MNISTDataset(
            os.path.join(path, 'train-images-idx3-ubyte'),
            os.path.join(path, 'train-labels-idx1-ubyte'),
            transform=train_transform
        )
        
        test_dataset = MNISTDataset(
            os.path.join(path, 't10k-images-idx3-ubyte'),
            os.path.join(path, 't10k-labels-idx1-ubyte'),
            transform=test_transform
        )
        
        # Create data loaders
        num_workers = min(4, os.cpu_count() or 1)
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Initialize model
        model = AdvancedMNISTModel(learning_rate=1e-3)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                monitor='val_acc',
                dirpath='checkpoints',
                filename='mnist-{epoch:02d}-{val_acc:.4f}',
                save_top_k=3,
                mode='max'
            ),
            EarlyStopping(
                monitor='val_acc',
                patience=5,
                mode='max'
            ),
            LearningRateMonitor(logging_interval='step')
        ]
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=30,
            accelerator='auto',
            devices=1,
            callbacks=callbacks,
            precision=16,
            gradient_clip_val=0.5,
            deterministic=True,
            accumulate_grad_batches=2
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.fit(model, train_loader, test_loader)
        
        # Test model
        logger.info("Testing model...")
        trainer.test(model, test_loader)
        
        # Save results
        os.makedirs('results', exist_ok=True)
        plot_confusion_matrix(
            confusion_matrix(model.val_acc.target, model.val_acc.preds),
            'results/confusion_matrix.png'
        )
        
        # Export model
        model = model.to('cpu')
        model.eval()
        
        os.makedirs('checkpoints', exist_ok=True)
        
        try:
            torch.save(model.state_dict(), 'checkpoints/mnist_advanced.pth')
            example_input = torch.randn(1, 1, 28, 28)
            traced_model = torch.jit.trace(model, example_input)
            torch.jit.save(traced_model, 'checkpoints/mnist_advanced_jit.pt')
            
            logger.info("\nTraining completed!")
            logger.info("\nFiles saved:")
            logger.info("1. checkpoints/mnist_advanced.pth - PyTorch state dict")
            logger.info("2. checkpoints/mnist_advanced_jit.pt - TorchScript model")
            logger.info("3. results/confusion_matrix.png - Confusion matrix plot")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nAn error occurred: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
