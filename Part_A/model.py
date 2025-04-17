import torch
import torch.nn as nn
import numpy as np

class CustomCNN(nn.Module):
    def __init__(self, config):
        super(CustomCNN, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 3
        img_size = 224
        
        # Get filter counts based on organization strategy
        if config['filter_organization'] == 'same':
            filters_per_layer = [config['num_filters']] * 5
        elif config['filter_organization'] == 'doubling':
            filters_per_layer = [config['num_filters'] * (2**i) for i in range(5)]
        elif config['filter_organization'] == 'halving':
            filters_per_layer = [config['num_filters'] // (2**i) for i in range(5)]
        
        # Create conv blocks
        for i in range(5):
            out_channels = filters_per_layer[i]
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=config['filter_size'], padding='same'),
                self._get_activation(config['activation']),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(out_channels) if config['use_batchnorm'] else nn.Identity(),
                nn.Dropout(config['dropout']) if config['use_dropout'] else nn.Identity()
            )
            self.layers.append(conv_block)
            in_channels = out_channels
            img_size //= 2
        
        self.flat_features = out_channels * (img_size) * (img_size)
        
        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(self.flat_features, config['dense_neurons']),
            self._get_activation(config['activation']),
            nn.Dropout(config['dropout']) if config['use_dropout'] else nn.Identity(),
            nn.Linear(config['dense_neurons'], config['num_classes'])
        )
    
    def _get_activation(self, activation_name):
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish()
        }
        return activations.get(activation_name, nn.ReLU())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, self.flat_features)
        x = self.dense(x)
        return x
