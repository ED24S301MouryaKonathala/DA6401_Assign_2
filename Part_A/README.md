# CNN Model Training on iNaturalist Dataset

This project implements a CNN model for image classification on the iNaturalist dataset. The implementation includes model training, hyperparameter tuning using wandb sweeps, and model evaluation.

## Project Structure
```
Part_A/
├── model.py           # CNN model architecture definition
├── dataset.py         # Dataset loading and preprocessing
├── train.py          # Training script with wandb sweeps
├── train_final.py    # Final model training script
├── test_final.py     # Testing script for final model
├── test_sweep_best_model_raw.py  # Testing script for sweep's best model
├── train_sweep_best_model.py # Extended training script for sweep's best model
├── test_sweep_retrained.py # Testing script of Extended trained sweep's best model
└── README.md         # Project documentation
```

## Requirements
- Python 3.8+
- PyTorch
- wandb
- numpy
- sklearn

## Setup
1. Install dependencies:
```bash
pip install torch wandb numpy sklearn
```

2. Configure wandb:
```bash
wandb login
```

3. Prepare dataset:
- Place the iNaturalist dataset in the parent directory
- Structure should be:
  ```
  nature_12K/
  ├── train/
  └── test/
  ```

## Training Process

### 1. Hyperparameter Sweep
Run hyperparameter sweep to find best configuration:
```bash
python train.py
```

This will:
- Perform Bayesian optimization sweep
- Test different model configurations
- Log results to wandb
- Save best model as 'best_model_{run_id}.pth'

### 2. Extended Training of Sweep's Best Model
Train the best model from sweep for additional epochs:
```bash
python train_test_sweep_best_model.py
```
Features:
- Loads best model from sweep
- Continues training for 25 epochs
- Full test evaluation
- Logs all metrics and visualizations to wandb

### 3. Final Model Training
Train the final model with best hyperparameters from scratch:
```bash
python train_final.py
```

Features:
- Uses best hyperparameters from sweep
- Trains for 25 epochs
- Saves best model as 'final_best_model.pth'

## Evaluation

### Test Final Model
```bash
python test_final.py
```

### Test Sweep's Best Model
```bash
python test_sweep_best_model_raw.py
```

Both testing scripts provide:
- Test accuracy metrics
- Confusion matrix
- 10×3 prediction grid visualization
- Results logged to wandb

## Model Architecture
- 5 convolution layers
- Each conv layer followed by:
  - Activation (configurable)
  - Max pooling
  - Batch normalization (optional)
  - Dropout (optional)
- One dense layer
- Output layer for classification

## Hyperparameters
Configurable parameters:
- Number of filters: [32, 64]
- Filter size: [3, 5]
- Activation functions: [ReLU, GELU, SiLU, Mish]
- Filter organization: [same, doubling, halving]
- Batch normalization: [True, False]
- Dropout rate: [0.2, 0.3]
- Data augmentation: [True, False]

## Results
Track results and visualizations on wandb:
- Training/validation metrics
- Confusion matrices
- Prediction grids
- Hyperparameter importance
- Model performance analysis

## License
Academic project - No license
