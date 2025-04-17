# EfficientNetV2-S Fine-tuning for iNaturalist Classification

Fine-tuning implementation of EfficientNetV2-S for the iNaturalist dataset classification task.

## Model Architecture
- Base: EfficientNetV2-S pretrained on ImageNet
- Modified final classifier for 10 classes
- Partial fine-tuning strategy:
  - Frozen: Up to stage 4
  - Trainable: Stage 5 and classifier

## Requirements
- PyTorch
- torchvision
- wandb
- numpy

## Setup
```bash
pip install torch torchvision wandb numpy
wandb login
```

## Directory Structure
```
Part_B/
├── model.py           # Fine-tuned EfficientNet implementation
├── dataset.py         # Data loading and preprocessing
├── train.py          # Training script
├── test.py           # Evaluation script
└── README.md         # Documentation
```

## Training
```bash
python train.py
```

## Testing
```bash
python test.py
```

## Features
- Proper layer freezing for transfer learning
- ImageNet standard preprocessing
- Training progress tracking
- Wandb integration for:
  - Training metrics
  - Confusion matrix
  - 10×3 prediction grid
  - Performance analysis

## Training Details
- Training data split: 80% train, 20% validation
- Complete test set for final evaluation
- Full training on train split
- Validation monitoring during training
- Best model saved based on validation accuracy

## Data Configuration
- Training split: 80% of training data
- Validation split: 20% of training data
- Test set: Complete separate test set
- Input size: 224×224
- ImageNet standard preprocessing

## Model Configuration
- Base model: EfficientNetV2-S (pretrained)
- Frozen layers: Up to stage 4
- Trainable layers: Stage 5 and classifier
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Learning rate: 1e-4
- Batch size: 32

## Logging & Visualization
- Training metrics (loss/accuracy)
- Validation metrics
- Test metrics
- Confusion matrix
- 10×3 prediction grid visualization
- Model performance analysis
