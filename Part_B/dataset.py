import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_V2_S_Weights
import os

def get_dataloaders(batch_size, dataset_path, test_mode=False):
    # Get EfficientNet's preprocessing
    preprocess = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
    
    if not test_mode:
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = preprocess
    
    val_transform = preprocess
    
    # Setup paths
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    
    print(f"Looking for dataset at: {dataset_path}")
    print(f"Looking for train directory at: {train_dir}")
    print(f"Looking for test directory at: {test_dir}")
    
    # Load datasets
    if not test_mode:
        # Load training data
        full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        
        # Calculate split sizes (80% train, 20% validation)
        total_size = len(full_train_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        # Split dataset
        train_dataset, val_dataset = random_split(
            full_train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update validation transform
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = None
    else:
        # Load test data only
        test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        train_loader = None
        val_loader = None
    
    return train_loader, val_loader, test_loader
