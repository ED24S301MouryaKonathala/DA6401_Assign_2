import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import numpy as np

DEFAULT_DATASET_PATH = os.path.normpath('D:\Mourya_Files\Deep Learning\Assign 2\nature_12K')

class iNaturalistDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Filter out non-directory items and .DS_Store
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d)) 
                             and d != '.DS_Store'])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            # Only process files, skip directories and .DS_Store
            for img_name in [f for f in os.listdir(class_path) 
                           if os.path.isfile(os.path.join(class_path, f))
                           and f != '.DS_Store']:
                self.images.append(os.path.join(class_path, img_name))
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a default image or skip
            random_idx = (idx + 1) % len(self)
            return self.__getitem__(random_idx)

def get_dataloaders(batch_size=32, use_augmentation=True, dataset_path=None, test_mode=False):
    # Normalize path separators
    if dataset_path is None:
        dataset_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nature_12k'))
    elif not os.path.isabs(dataset_path):
        dataset_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_path))
    
    train_dir = os.path.normpath(os.path.join(dataset_path, 'train'))
    test_dir = os.path.normpath(os.path.join(dataset_path, 'test'))
    
    # Print attempted paths for debugging
    print(f"Looking for dataset at: {dataset_path}")
    print(f"Looking for train directory at: {train_dir}")
    print(f"Looking for test directory at: {test_dir}")
    
    # Check each path separately
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Main dataset directory not found at: {dataset_path}")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")
        
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")
    
    # Data augmentation with size handling
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Force resize to square
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) if use_augmentation else transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if test_mode:
        # Only create test loader when in test mode
        test_dataset = iNaturalistDataset(root_dir=test_dir, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return None, None, test_loader
    
    # Create datasets
    train_dataset = iNaturalistDataset(root_dir=train_dir, transform=train_transform)
    
    # Calculate class distribution for stratified split
    class_counts = {}
    for label in train_dataset.labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Ensure balanced validation split
    train_idx, val_idx = train_test_split(
        np.arange(len(train_dataset)),
        test_size=0.2,  # 20% validation split
        stratify=train_dataset.labels,
        random_state=42
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx)
    )
    
    val_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx)
    )
    
    return train_loader, val_loader, None
