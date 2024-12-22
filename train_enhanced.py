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
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
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
        
        # Classification head
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

def train_model(model, train_loader, test_loader, epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    
    # Training metrics
    best_acc = 0
    train_losses = []
    test_accuracies = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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
                images, labels = images.to(device), labels.to(device)
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, 'best_mnist_model.pth')
            logger.info(f'New best model saved with accuracy: {accuracy:.2f}%')
    
    total_time = time.time() - start_time
    logger.info(f'\nTraining completed in {total_time/60:.2f} minutes')
    logger.info(f'Best test accuracy: {best_acc:.2f}%')
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('training_progress.png')
    plt.close()
    
    return model, best_acc

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
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
    
    # Create data loaders
    train_loader = DataLoader(
        trainset, batch_size=128, shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        testset, batch_size=128, shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
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
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': best_acc,
    }, 'final_mnist_model.pth')
    
    # Save TorchScript model for deployment
    model.eval()
    example = torch.randn(1, 1, 28, 28).to(device)
    traced_model = torch.jit.trace(model, example)
    torch.jit.save(traced_model, 'mnist_model_scripted.pt')
    
    logger.info('\nTraining completed! Files saved:')
    logger.info('1. best_mnist_model.pth - Best checkpoint during training')
    logger.info('2. final_mnist_model.pth - Final model state')
    logger.info('3. mnist_model_scripted.pt - TorchScript model for deployment')
    logger.info('4. training_progress.png - Training progress plot')

if __name__ == '__main__':
    main()
