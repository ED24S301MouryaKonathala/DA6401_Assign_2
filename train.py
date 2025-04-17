import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CustomCNN
from dataset import get_dataloaders
import torch.optim as optim
import math
import os
from pathlib import Path

def train_model(config=None):
    with wandb.init(config=config, reinit=True):
        config = wandb.config
        print(f"\n{'='*80}")
        print(f"Starting sweep run {wandb.run.name} ({wandb.run.id})")
        print(f"Configuration:\n{config}")
        print(f"{'='*80}\n")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Get dataloaders with dataset path
        train_loader, val_loader, _ = get_dataloaders(
            batch_size=config.batch_size,
            use_augmentation=config.use_augmentation,
            dataset_path=config.dataset_path,
            test_mode=False
        )
        
        # Log input image dimensions
        sample_batch = next(iter(train_loader))
        wandb.config.update({"input_dims": list(sample_batch[0].shape[1:])})
        
        # Initialize model and tracking variables
        model = CustomCNN(config).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        best_val_accuracy = 0.0
        best_model_state = None

        # Training loop
        for epoch in range(config.epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            total_train = 0
            
            # Progress tracking for training
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
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
                    
                    # Print batch progress
                    if (batch_idx + 1) % 10 == 0:
                        print(f"Epoch [{epoch+1}/{config.epochs}] "
                              f"Batch [{batch_idx+1}/{len(train_loader)}] "
                              f"Loss: {loss.item():.4f}")
                    
                    wandb.log({
                        "train_loss": loss.item(),
                        "epoch": epoch
                    })
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100. * train_correct / total_train
            
            # Log training metrics to wandb
            wandb.log({
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "epoch": epoch
            })
            
            # Validation metrics
            val_loss = 0
            correct = 0
            total_val = 0
            model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total_val += target.size(0)
            
            val_loss /= len(val_loader)
            accuracy = correct / total_val
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Training Loss: {epoch_train_loss:.4f} | Training Acc: {epoch_train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {accuracy*100:.2f}%")
            print(f"{'-'*80}\n")
            
            # Save best model
            if accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
                best_model_state = model.state_dict().copy()
            
            wandb.log({
                "val_loss": val_loss,
                "val_accuracy": accuracy,
                "epoch": epoch
            })

        # Save best model - modified to handle Windows path issues
        best_model_path = Path(f"best_model_{wandb.run.id}.pth")
        torch.save(best_model_state, best_model_path)
        try:
            wandb.save(str(best_model_path))
        except OSError:
            print(f"Warning: Could not create symlink for {best_model_path}, but model was saved.")
        
        # Save accuracy for later comparison
        with open("sweep_results.txt", "a") as f:
            f.write(f"{wandb.run.id},{best_val_accuracy},{str(best_model_path)}\n")

        print(f"\nCompleted sweep run {wandb.run.name}")
        print(f"Best validation accuracy: {best_val_accuracy*100:.2f}%")
        print(f"{'='*80}\n")

def evaluate_best_model(best_config):
    """Evaluate the best model configuration on the test set"""
    print("\nEvaluating best model on test set...")
    with wandb.init(config=best_config, name="best_model_test", reinit=True):
        config = wandb.config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get only test loader
        _, _, test_loader = get_dataloaders(
            batch_size=config.batch_size,
            dataset_path=config.dataset_path,
            test_mode=True
        )
        
        model = CustomCNN(config).to(device)
        model.load_state_dict(torch.load(f"best_model_{wandb.run.id}.pth"))
        
        criterion = nn.CrossEntropyLoss()
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        test_accuracy = correct / len(test_loader.dataset) * 100  # Convert to percentage
        
        print(f"\nTest Set Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "final_test_accuracy": test_accuracy
        })

def find_best_model():
    """Find the best model from all sweep runs"""
    best_accuracy = 0
    best_model_path = None
    best_run_id = None
    
    try:
        with open("sweep_results.txt", "r") as f:
            for line in f:
                run_id, accuracy, model_path = line.strip().split(",")
                accuracy = float(accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_path = model_path
                    best_run_id = run_id
    except FileNotFoundError:
        print("No sweep results found!")
        return None
    
    return best_run_id, best_model_path, best_accuracy

# Wandb sweep configuration with optimized search space
sweep_config = {
    'method': 'bayes',  # Bayesian optimization for more efficient hyperparameter search vs random/grid search
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        # Limited to smaller filter counts to prevent overfitting and reduce memory usage
        # Removed 128 filters as it led to model complexity without proportional gains
        'num_filters': {'values': [64, 32]},
        
        # Smaller filters capture fine-grained features better
        # Removed 7x7 as it loses detail and increases params without benefit
        'filter_size': {'values': [3, 5]},
        
        # Different filter organization strategies to test capacity scaling
        'filter_organization': {'values': ['doubling', 'same', 'halving']},
        
        # Modern activation functions known to work well with CNNs
        # Removed tanh/sigmoid due to vanishing gradient issues
        'activation': {'values': ['mish', 'silu', 'gelu', 'relu']},
        
        # Moderate dense layer sizes to balance capacity vs overfitting
        # Removed 2048 as it led to overfitting on this dataset size
        'dense_neurons': {'values': [1024, 512]},
        
        # Log-uniform distribution for learning rate - better coverage of small values
        # Range chosen based on typical Adam optimizer performance
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': math.log(1e-4),
            'max': math.log(3e-3)
        },
        
        # Larger batch size first for more stable gradients
        'batch_size': {'values': [64, 32]},
        
        # Essential for stable CNN training - always enabled
        'use_batchnorm': {'value': True},
        
        # Dropout is crucial for regularization - always enabled
        'use_dropout': {'value': True},
        
        # Moderate dropout rates - higher values hurt convergence
        'dropout': {'values': [0.2, 0.3]},
        
        # Data augmentation essential for natural image classification
        'use_augmentation': {'value': True},
        
        # Dynamic path construction for better portability
        'dataset_path': {'value': os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nature_12K'))},
        
        # Reduced from 30 to 15 epochs as most models converged earlier
        'epochs': {'value': 15}
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3  # Stop poorly performing runs early to save compute
    }
}

if __name__ == "__main__":
    wandb.login()
    
    # Create or clear sweep results file
    with open("sweep_results.txt", "w") as f:
        f.write("run_id,accuracy,model_path\n")
    
    print("\nStarting hyperparameter sweep")
    print(f"{'='*80}")
    print("Will run 20 different configurations")
    print(f"{'='*80}\n")
    
    # Run hyperparameter sweep
    sweep_id = wandb.sweep(sweep_config, project="Assign_2_Part_A_iNaturalist_CNN")
    wandb.agent(sweep_id, train_model, count=20)
    
    # Find best model from sweep results
    best_run_id, best_model_path, best_accuracy = find_best_model()
    print(f"\nBest model from all sweeps:")
    print(f"Run ID: {best_run_id}")
    print(f"Validation Accuracy: {best_accuracy*100:.2f}%")
    print(f"Model Path: {best_model_path}")
    
    # Get config of best run
    api = wandb.Api()
    best_run = api.run(f"mourya001-indian-institute-of-technology-madras/Assign_2_Part_A_iNaturalist_CNN/runs/{best_run_id}")
    
    # Evaluate best model on test set
    evaluate_best_model(best_run.config)
