import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CustomCNN
from dataset import get_dataloaders
import torch.optim as optim
import os
import numpy as np

# Best hyperparameters from sweep
config = {
    'activation': 'relu',
    'batch_size': 32,
    'dense_neurons': 512,
    'dropout': 0.2,
    'filter_organization': 'same',
    'filter_size': 3,
    'learning_rate': 0.0003,
    'num_filters': 64,
    'use_batchnorm': True,
    'use_dropout': True,
    'use_augmentation': True,
    'dataset_path': os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nature_12K')),
    'epochs': 25  # Fixed 25 epochs for final training
}

def train_final_model():
    with wandb.init(project="Assign_2_Part_A_iNaturalist_CNN_Final", name="final_best_model", config=config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Get dataloaders - set test_mode to False to get all loaders
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=config['batch_size'],
            use_augmentation=config['use_augmentation'],
            dataset_path=config['dataset_path'],
            test_mode=False  # Added this parameter
        )
        
        # Verify test_loader is not None
        if test_loader is None:
            print("Warning: test_loader is None, getting test loader separately")
            _, _, test_loader = get_dataloaders(
                batch_size=config['batch_size'],
                use_augmentation=False,  # No augmentation for test
                dataset_path=config['dataset_path'],
                test_mode=True
            )
        
        # Get actual class names from dataset and update config
        class_names = train_loader.dataset.classes
        model_config = config.copy()
        model_config['num_classes'] = len(class_names)
        
        # Initialize model with config object
        model = CustomCNN(model_config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        best_val_accuracy = 0.0
        best_model_state = None
        
        # Training loop
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
                
                # Log batch metrics to wandb
                wandb.log({
                    "batch_train_loss": loss.item(),
                    "batch": batch_idx + epoch * len(train_loader)
                })
                
                # Print batch progress
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
            
            print(f"\nStarting validation for epoch {epoch+1}...")
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    total_val += target.size(0)
                    
                    # Log validation batch metrics
                    wandb.log({
                        "batch_val_loss": criterion(output, target).item(),
                        "batch": batch_idx + epoch * len(val_loader)
                    })
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / total_val
            
            # Log epoch metrics to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Training Loss: {train_loss:.4f} | Training Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"{'-'*80}\n")

            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = model.state_dict().copy()
        
        # Save final best model
        torch.save(best_model_state, "final_best_model.pth")
        print(f"\nTraining completed. Best validation accuracy: {best_val_accuracy:.2f}%")
        print(f"Best model saved as: final_best_model.pth")

if __name__ == "__main__":
    wandb.login()
    train_final_model()
