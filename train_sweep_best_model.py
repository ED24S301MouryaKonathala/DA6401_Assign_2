import wandb
import torch
import torch.nn as nn
from model import CustomCNN
from dataset import get_dataloaders
import torch.optim as optim
import os

# Configuration matching the sweep's best model
config = {
    'activation': 'relu',
    'batch_size': 32,
    'dense_neurons': 512,
    'dropout': 0.2,
    'filter_organization': 'same',
    'filter_size': 5,  # Important: This matches the sweep's best model
    'learning_rate': 0.0003,
    'num_filters': 64,
    'use_batchnorm': True,
    'use_dropout': True,
    'use_augmentation': True,
    'dataset_path': os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nature_12K')),
    'epochs': 25  # Extended training for 25 epochs
}

def train_sweep_model():
    with wandb.init(project="Assign_2_Part_A_iNaturalist_CNN_Final", name="sweep_model_extended_training", config=config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Get dataloaders
        train_loader, val_loader, _ = get_dataloaders(
            batch_size=config['batch_size'],
            use_augmentation=config['use_augmentation'],
            dataset_path=config['dataset_path'],
            test_mode=False
        )
        
        # Setup model with best sweep configuration
        class_names = train_loader.dataset.classes
        model_config = config.copy()
        model_config['num_classes'] = len(class_names)
        
        # Initialize model and load sweep's best weights
        model = CustomCNN(model_config).to(device)
        model.load_state_dict(torch.load("best_model_bzndq3jx.pth"))
        print("Loaded sweep's best model")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        best_val_accuracy = 0.0
        best_model_state = None
        
        # Extended training loop
        for epoch in range(config['epochs']):
            model.train()
            train_loss = 0
            train_correct = 0
            total_train = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                total_train += target.size(0)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{config['epochs']}] "
                          f"Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}")
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / total_train
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            total_val = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    total_val += target.size(0)
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / total_val
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = model.state_dict().copy()
        
        # Save the best model
        torch.save(best_model_state, "sweep_best_model_retrained.pth")
        print(f"\nSaved best model with validation accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    wandb.login()
    train_sweep_model()
