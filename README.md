# DA6401_Assign_2
# Deep Learning Assignment 2
## Part A and Part B files in PartA and PartB branches.
This repository contains implementations of deep learning models for the iNaturalist dataset classification task, exploring both custom CNN development and transfer learning approaches.

---

## Overview

The assignment implements and compares two different approaches to image classification:
- Part A: Custom CNN implementation with hyperparameter optimization
- Part B: Transfer learning using EfficientNetV2-S

Dataset: iNaturalist dataset with image classification task

---

## Part A: Custom CNN Implementation

Part A implements a custom CNN architecture with hyperparameter optimization:

Architecture Details:
- 5 convolution layers with configurable parameters
- Configurable activation functions (ReLU, GELU, SiLU, Mish)
- Max pooling after each conv layer
- Optional batch normalization and dropout
- Dense layer followed by output layer

Key Features:
- Wandb integration for experiment tracking
- Hyperparameter optimization using Bayesian search
- Configurable parameters include:
  - Filter counts: [32, 64]
  - Filter sizes: [3, 5]
  - Filter organization strategies
  - Dropout rates: [0.2, 0.3]
  - Data augmentation options

---

## Part B: EfficientNetV2-S Transfer Learning

Part B utilizes transfer learning with EfficientNetV2-S:

Implementation Details:
- Base: EfficientNetV2-S pretrained on ImageNet
- Fine-tuning strategy:
  - Frozen: Layers up to stage 4
  - Trainable: Stage 5 and classifier
- Modified classifier for iNaturalist classes

Training Configuration:
- Input size: 224×224
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 32
- ImageNet standard preprocessing

---

## Repository Structure

```
.
├── part_a/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── train_final.py
│   ├── test_final.py
│   └── README.md
├── part_b/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── test.py
│   └── README.md
└── README.md
```

---

## Requirements

```python
torch>=1.8.0
torchvision>=0.9.0
wandb
numpy
sklearn
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Place the iNaturalist dataset in the following structure:
```
nature_12K/
├── train/
└── test/
```

---

## Running the Code

### Part A:
```bash
cd part_a
# Run hyperparameter sweep
python train.py
# Train final model
python train_final.py
# Evaluate model
python test_final.py
```

### Part B:
```bash
cd part_b
# Train model
python train.py
# Evaluate model
python test.py
```

---

## Results & Visualization

Both implementations provide:
- Training/validation metrics
- Confusion matrices
- 10×3 prediction grid visualizations
- Model performance analysis

Results are logged to Weights & Biases for detailed analysis and comparison.

---

## License

This project is for academic purposes only.

---

## Acknowledgments

- iNaturalist dataset providers
- PyTorch framework team
- Weights & Biases for experiment tracking

