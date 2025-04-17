import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class FineTunedEfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load pretrained model
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        # Freeze layers up to stage 4
        for name, param in self.model.named_parameters():
            if 'blocks.6' not in name and 'classifier' not in name:  # stage 5 starts at blocks.6
                param.requires_grad = False
        
        # Modify classifier for 10 classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
