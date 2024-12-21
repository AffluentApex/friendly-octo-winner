import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from models import BasicCNN, ResNet, VGGNet
from training_utils import get_data_transforms
from tqdm import tqdm
import time

def train_model(model, train_loader, test_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Calculate total steps for progress bar
    total_batches = len(train_loader)
    total_steps = epochs * total_batches
    
    print("\nTraining Progress:")
    global_progress = tqdm(total=total_steps, desc="Overall Progress", position=0)
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_progress = tqdm(total=total_batches, desc=f"Epoch {epoch + 1}/{epochs}", 
                            position=1, leave=False)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bars
            epoch_progress.update(1)
            global_progress.update(1)
            
            # Update progress bar description with current metrics
            if batch_idx % 50 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                epoch_progress.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
        
        epoch_progress.close()
        
        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        test_progress = tqdm(total=len(test_loader), desc="Testing", position=1, leave=False)
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
                test_progress.update(1)
        
        test_accuracy = 100. * test_correct / test_total
        test_progress.close()
        
        print(f'\nEpoch {epoch + 1} Test Accuracy: {test_accuracy:.2f}%')
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f'New best accuracy: {best_accuracy:.2f}%')
    
    global_progress.close()
    return model

def train_all_models():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data transforms
    train_transform, test_transform = get_data_transforms()

    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                 transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False,
                                transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Create model instances
    models = {
        'basic_cnn': BasicCNN(),
        'resnet': ResNet(),
        'vgg': VGGNet()
    }

    # Create checkpoints directory
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # Train each model
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print(f"{'='*50}")
        
        model = model.to(device)
        
        # Train model
        start_time = time.time()
        model = train_model(model, train_loader, test_loader, device)
        training_time = time.time() - start_time
        
        # Save model
        torch.save(model.state_dict(), f'checkpoints/{name}_best.pth')
        print(f"\nSaved {name} model")
        print(f"Training time: {training_time/60:.2f} minutes")
        print(f"{'='*50}\n")

if __name__ == '__main__':
    train_all_models()
