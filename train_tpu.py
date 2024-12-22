import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import time
import logging
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class EnhancedMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(model, train_loader, test_loader, epochs=15, device=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    
    # Training metrics
    best_acc = 0
    test_accuracies = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            xm.optimizer_step(optimizer)
            scheduler.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}, '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}')
                running_loss = 0.0
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        logger.info(f'Epoch [{epoch+1}/{epochs}], '
              f'Test Loss: {test_loss/len(test_loader):.4f}, '
              f'Test Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }
            xm.save(checkpoint, 'best_mnist_model.pth')
            logger.info(f'New best model saved with accuracy: {accuracy:.2f}%')
    
    total_time = time.time() - start_time
    logger.info(f'\nTraining completed in {total_time/60:.2f} minutes')
    logger.info(f'Best test accuracy: {best_acc:.2f}%')
    
    # Plot training curves (only on main process)
    if xm.is_master_ordinal():
        plt.figure(figsize=(10, 5))
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig('training_progress.png')
        plt.close()
    
    return model, best_acc

def _mp_fn(index):
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Get TPU device
    device = xm.xla_device()
    logger.info(f'Using device: {device}')
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    logger.info('Loading MNIST dataset...')
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders optimized for TPU
    train_loader = pl.MpDeviceLoader(
        DataLoader(
            trainset,
            batch_size=1024,  # Larger batch size for TPU
            shuffle=True,
            num_workers=4,
            drop_last=True
        ),
        device
    )
    
    test_loader = pl.MpDeviceLoader(
        DataLoader(
            testset,
            batch_size=1024,
            shuffle=False,
            num_workers=4,
            drop_last=True
        ),
        device
    )
    
    # Initialize model
    model = EnhancedMNISTNet().to(device)
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Train model
    logger.info('Starting training...')
    model, best_acc = train_model(
        model, train_loader, test_loader,
        epochs=15, device=device
    )
    
    # Save final model (only on main process)
    if xm.is_master_ordinal():
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({
            'model_state_dict': state_dict,
            'accuracy': best_acc,
        }, 'final_mnist_model.pth')
        
        # Save TorchScript model for deployment
        model.cpu().eval()
        example = torch.randn(1, 1, 28, 28)
        traced_model = torch.jit.trace(model, example)
        torch.jit.save(traced_model, 'mnist_model_scripted.pt')
        
        logger.info('\nTraining completed! Files saved:')
        logger.info('1. best_mnist_model.pth - Best checkpoint during training')
        logger.info('2. final_mnist_model.pth - Final model state')
        logger.info('3. mnist_model_scripted.pt - TorchScript model for deployment')
        logger.info('4. training_progress.png - Training progress plot')

if __name__ == '__main__':
    xmp.spawn(_mp_fn, nprocs=8)  # Start 8 TPU processes
